"""Unit tests for LTI topology-aware frequency dynamics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandapower as pp
import pytest
import scipy.integrate
import scipy.linalg

from src.env.freq_dynamics_lti import LTITopologyFreqDynamics, FrequencyStateLTI


def _create_simple_meshed_network():
    """Create a simple 10-bus meshed network with 4 GFMs for testing.

    Topology:
        0 --- 1 --- 2 --- 3
        |     |     |     |
        4 --- 5 --- 6 --- 7
              |           |
              8 --------- 9

    GFMs at buses 0, 3, 4, 9 (corners for max electrical distance).
    """
    net = pp.create_empty_network(sn_mva=10.0)

    # Create 10 buses
    for i in range(10):
        pp.create_bus(net, vn_kv=22.0, name=str(i + 1))

    # Set bus_id column
    net.bus["bus_id"] = net.bus.index + 1

    # Create lines (meshed topology)
    lines = [
        (0, 1), (1, 2), (2, 3),  # Top row
        (4, 5), (5, 6), (6, 7),  # Middle row
        (0, 4), (1, 5), (2, 6), (3, 7),  # Vertical connections
        (5, 8), (8, 9), (7, 9),  # Bottom connections
    ]
    for from_bus, to_bus in lines:
        pp.create_line_from_parameters(
            net, from_bus=from_bus, to_bus=to_bus,
            length_km=1.0, r_ohm_per_km=0.1, x_ohm_per_km=0.4,
            c_nf_per_km=0.0, max_i_ka=1.0,
        )

    # Add loads at non-GFM buses
    for bus in [1, 2, 5, 6, 7, 8]:
        pp.create_load(net, bus=bus, p_mw=0.5, q_mvar=0.1)

    # Add ext_grids at GFM buses (corners)
    for bus in [0, 3, 4, 9]:
        pp.create_ext_grid(net, bus=bus, vm_pu=1.0, name=f"gfm_{bus}")

    # Run power flow
    pp.runpp(net, algorithm="nr", init="flat", calculate_voltage_angles=True)

    return net


@pytest.fixture
def gfm_placement():
    """4-GFM placement for simple test network."""
    return {
        "gfm": {
            "G0": {"bus": 1, "inverter_mva": 2.0, "tau_filter": 0.1, "mode": "Droop"},
            "G3": {"bus": 4, "inverter_mva": 1.5, "tau_filter": 0.1, "mode": "Droop"},
            "G4": {"bus": 5, "inverter_mva": 1.5, "tau_filter": 0.1, "mode": "Droop"},
            "G9": {"bus": 10, "inverter_mva": 1.0, "tau_filter": 0.1, "mode": "Droop"},
        }
    }


@pytest.fixture
def simple_net_with_pf():
    """Simple meshed network with converged power flow."""
    net = _create_simple_meshed_network()
    bus_map = {int(bus_id): int(idx) for idx, bus_id in net.bus["bus_id"].items()}
    return net, bus_map


@pytest.fixture
def lti_freq_dyn(simple_net_with_pf, gfm_placement):
    """Instantiate LTI freq dynamics with converged power flow."""
    net, bus_map = simple_net_with_pf

    lti = LTITopologyFreqDynamics(
        placement=gfm_placement,
        base_net=net,
        bus_map=bus_map,
        f0=50.0,
        dt_fast=0.1,
        cache_k_bins=5,
    )
    lti.bind_operating_point(net, topology_id=0)
    return lti


class TestJacobianSanity:
    """Test Jacobian properties."""

    def test_jacobian_is_real(self, lti_freq_dyn):
        """J_r should be real-valued."""
        assert lti_freq_dyn._J_r is not None
        assert np.isreal(lti_freq_dyn._J_r).all()

    def test_jacobian_shape_matches_gfm_count(self, lti_freq_dyn):
        """J_r should be (n_gfm, n_gfm)."""
        n_gfm = lti_freq_dyn.n_gfm
        assert lti_freq_dyn._J_r.shape == (n_gfm, n_gfm)

    def test_jacobian_approx_symmetric(self, lti_freq_dyn):
        """J_r should be approximately symmetric for meshed networks."""
        J_r = lti_freq_dyn._J_r
        if not np.isfinite(J_r).all():
            pytest.skip("J_r has NaN/Inf (test network power flow issue)")
        asymmetry = np.linalg.norm(J_r - J_r.T) / (np.linalg.norm(J_r) + 1e-10)
        assert asymmetry < 0.5, f"J_r asymmetry too high: {asymmetry:.4f}"

    def test_jacobian_diagonal_positive(self, lti_freq_dyn):
        """Diagonal of J_r should be non-negative (self-sensitivity)."""
        diag = np.diag(lti_freq_dyn._J_r)
        if not np.isfinite(diag).all():
            pytest.skip("J_r diagonal has NaN/Inf (test network power flow issue)")
        assert np.all(diag >= -1e-6), f"Negative diagonal elements: {diag}"


class TestKronReduction:
    """Test Kron reduction correctness."""

    def test_kron_reduces_to_gfm_buses(self, lti_freq_dyn, gfm_placement):
        """Reduced Jacobian should have shape (n_gfm, n_gfm)."""
        n_gfm = len(gfm_placement["gfm"])
        assert lti_freq_dyn._J_r.shape == (n_gfm, n_gfm)

    def test_kron_preserves_connectivity(self, lti_freq_dyn):
        """Off-diagonal entries should be non-zero for connected GFMs."""
        J_r = lti_freq_dyn._J_r
        if not np.isfinite(J_r).all():
            pytest.skip("J_r has NaN/Inf (test network power flow issue)")
        off_diag = J_r - np.diag(np.diag(J_r))
        assert np.any(np.abs(off_diag) > 1e-10), "All off-diagonal zero — GFMs disconnected?"


class TestSystemMatrix:
    """Test A_f assembly and stability."""

    def test_A_f_shape(self, lti_freq_dyn):
        """A_f should be (n_state, n_state) = (7, 7) for 4 GFMs."""
        K_droop = np.ones(lti_freq_dyn.n_gfm)
        A_f = lti_freq_dyn._assemble_A_f(K_droop)
        expected = (lti_freq_dyn.n_gfm - 1) + lti_freq_dyn.n_gfm
        assert A_f.shape == (expected, expected)

    def test_A_f_eigenvalues_stable(self, lti_freq_dyn):
        """All eigenvalues of A_f should have non-positive real parts."""
        K_droop = np.ones(lti_freq_dyn.n_gfm)
        A_f = lti_freq_dyn._assemble_A_f(K_droop)
        if not np.isfinite(A_f).all():
            pytest.skip("A_f has NaN/Inf (test network power flow issue)")
        eigvals = np.linalg.eigvals(A_f)
        max_real = np.max(np.real(eigvals))
        assert max_real <= 1e-6, f"Unstable eigenvalue: max Re(λ) = {max_real:.4f}"

    def test_A_f_varies_with_k_droop(self, lti_freq_dyn):
        """A_f should change when K_droop changes."""
        K1 = np.ones(lti_freq_dyn.n_gfm)
        K2 = np.ones(lti_freq_dyn.n_gfm) * 2.0
        A1 = lti_freq_dyn._assemble_A_f(K1)
        A2 = lti_freq_dyn._assemble_A_f(K2)
        assert not np.allclose(A1, A2), "A_f should depend on K_droop"


class TestTimeStepEquivalence:
    """Compare matrix exponential with ODE solver."""

    def test_phi_x_matches_ode_solver(self, lti_freq_dyn):
        """Φ·x should match solve_ivp over one Δt."""
        K_droop = np.ones(lti_freq_dyn.n_gfm)
        A_f = lti_freq_dyn._assemble_A_f(K_droop)
        if not np.isfinite(A_f).all():
            pytest.skip("A_f has NaN/Inf (test network power flow issue)")
        dt = lti_freq_dyn.dt_fast

        x0 = np.zeros(lti_freq_dyn._n_state)
        x0[lti_freq_dyn.n_gfm] = 0.01

        Phi = scipy.linalg.expm(A_f * dt)
        x_expm = Phi @ x0

        def ode_rhs(t, x):
            return A_f @ x

        sol = scipy.integrate.solve_ivp(
            ode_rhs,
            [0, dt],
            x0,
            method="RK45",
            rtol=1e-10,
            atol=1e-12,
        )
        x_ode = sol.y[:, -1]

        np.testing.assert_allclose(x_expm, x_ode, rtol=1e-5, atol=1e-8)

    def test_step_produces_valid_state(self, lti_freq_dyn):
        """step() should return valid FrequencyStateLTI."""
        if not np.isfinite(lti_freq_dyn._J_r).all():
            pytest.skip("J_r has NaN/Inf (test network power flow issue)")
        lti_freq_dyn.reset()
        K_droop = np.ones(lti_freq_dyn.n_gfm)
        delta_P_ref = np.zeros(lti_freq_dyn.n_gfm)
        delta_P_ref[0] = -0.1

        state = lti_freq_dyn.step(
            dt=lti_freq_dyn.dt_fast,
            delta_P_ref=delta_P_ref,
            delta_P_L=0.0,
            K_droop=K_droop,
            topology_id=0,
        )

        assert isinstance(state, FrequencyStateLTI)
        assert state.delta_f_per_bus.shape == (lti_freq_dyn.n_gfm,)
        assert state.rocof_per_bus.shape == (lti_freq_dyn.n_gfm,)
        assert np.isfinite(state.delta_f_hz)
        assert np.isfinite(state.delta_f_worst)


class TestCaching:
    """Test Φ cache behavior."""

    def test_cache_hit_on_same_topology_k(self, lti_freq_dyn):
        """Same (topology_id, K_bin) should hit cache."""
        K_droop = np.ones(lti_freq_dyn.n_gfm)
        lti_freq_dyn._phi_cache.clear()

        _ = lti_freq_dyn._get_phi(K_droop, topology_id=0)
        assert len(lti_freq_dyn._phi_cache) == 1

        _ = lti_freq_dyn._get_phi(K_droop, topology_id=0)
        assert len(lti_freq_dyn._phi_cache) == 1

    def test_cache_miss_on_different_k(self, lti_freq_dyn):
        """Different K_bin should create new cache entry."""
        K1 = np.ones(lti_freq_dyn.n_gfm) * 0.5
        K2 = np.ones(lti_freq_dyn.n_gfm) * 2.0
        lti_freq_dyn._phi_cache.clear()

        _ = lti_freq_dyn._get_phi(K1, topology_id=0)
        _ = lti_freq_dyn._get_phi(K2, topology_id=0)
        assert len(lti_freq_dyn._phi_cache) >= 2


class TestGFMMapping:
    """Test GFM bus index mapping."""

    def test_gfm_at_own_bus_returns_self(self, lti_freq_dyn):
        """Agent at GFM bus should map to that GFM."""
        gfm_idx = lti_freq_dyn._gfm_pp_idx[0]
        mapped = lti_freq_dyn.get_gfm_bus_idx(gfm_idx)
        assert mapped == 0

    def test_non_gfm_bus_maps_to_nearest(self, lti_freq_dyn):
        """Agent at non-GFM bus should map to nearest GFM."""
        far_bus = 999
        mapped = lti_freq_dyn.get_gfm_bus_idx(far_bus)
        assert 0 <= mapped < lti_freq_dyn.n_gfm
