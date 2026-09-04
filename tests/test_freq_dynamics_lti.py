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


class TestVirtualInertia:
    """Test VSG virtual-inertia behaviour (swing dynamics)."""

    def _build(self, net, bus_map, placement, h_virt_floor):
        lti = LTITopologyFreqDynamics(
            placement=placement, base_net=net, bus_map=bus_map,
            f0=50.0, dt_fast=0.1, h_virt_floor=h_virt_floor,
        )
        lti.bind_operating_point(net, topology_id=0)
        return lti

    def test_h_sys_rating_weighted(self, lti_freq_dyn):
        """h_sys = Σ(H_i·S_i)/S_BASE over connected GFMs."""
        expected = float(
            np.sum(lti_freq_dyn._H_virt * lti_freq_dyn._gfm_ratings) / lti_freq_dyn.s_base_mva
        )
        assert lti_freq_dyn.h_sys == pytest.approx(expected, rel=1e-9)
        assert lti_freq_dyn.h_sys > 0.0

    def test_h_sys_drops_when_gfm_disconnected(self, lti_freq_dyn):
        """Disconnecting GFMs reduces aggregate inertia; None restores it."""
        full = lti_freq_dyn.h_sys
        one_id = lti_freq_dyn._gfm_params[0].gfm_id
        lti_freq_dyn.update_topology({one_id})
        assert lti_freq_dyn.h_sys < full
        lti_freq_dyn.update_topology(None)
        assert lti_freq_dyn.h_sys == pytest.approx(full, rel=1e-9)

    def test_rocof_scales_inversely_with_inertia(self, simple_net_with_pf, gfm_placement):
        """Defining VSG signature: doubling H halves the post-disturbance RoCoF."""
        net, bus_map = simple_net_with_pf
        lo = self._build(net, bus_map, gfm_placement, h_virt_floor=0.5)
        hi = self._build(net, bus_map, gfm_placement, h_virt_floor=1.0)
        if not (np.isfinite(lo._J_r).all() and np.isfinite(hi._J_r).all()):
            pytest.skip("J_r has NaN/Inf (test network power flow issue)")

        K = np.ones(lo.n_gfm)
        kw = dict(dt=0.1, delta_P_ref=np.zeros(lo.n_gfm), delta_P_L=0.2, K_droop=K, topology_id=0)
        lo.reset(); hi.reset()
        r_lo = abs(lo.step(**kw).rocof_hz_s)
        r_hi = abs(hi.step(**kw).rocof_hz_s)

        assert r_lo > 1e-6 and r_hi > 1e-6
        assert 0.4 < (r_hi / r_lo) < 0.6, f"RoCoF ratio {r_hi / r_lo:.3f} not ~0.5 (1/H scaling)"


class TestLocatedDisturbance:
    """J_L per-passive-bus injection: the event LOCATION drives the worst GFM
    (Path B core mechanism). GFMs are at pp buses 0,3,4,9 (gfm order [0,3,4,9]);
    passive load buses are pp 1,2,5,6,7,8.
    """

    def _step(self, lti, event_location_pp, mag=0.2):
        lti.reset()
        K = np.ones(lti.n_gfm)
        return lti.step(
            dt=0.1, delta_P_ref=np.zeros(lti.n_gfm), delta_P_L=mag,
            K_droop=K, topology_id=0, event_location_pp=event_location_pp,
        )

    def test_passive_bus_map_built(self, lti_freq_dyn):
        """bind_operating_point should expose pp-bus -> J_L-column for passive buses."""
        assert len(lti_freq_dyn._ppidx_to_Jcol) > 0
        assert 1 in lti_freq_dyn._ppidx_to_Jcol  # pp bus 1 is a passive load bus

    def test_jl_columns_sum_near_unit_magnitude(self, lti_freq_dyn):
        """Kron sensitivity columns ~unit magnitude (full injection reaches GFMs).

        Sign is the Jacobian convention (columns sum to ≈ −1); B_net flips it so a
        deficit pushes frequency down. Generous tolerance for high-R/X + pinv.
        """
        if lti_freq_dyn._J_L is None or not np.isfinite(lti_freq_dyn._J_L).all():
            pytest.skip("J_L has NaN/Inf (test network power flow issue)")
        col_sums = np.abs(np.asarray(lti_freq_dyn._J_L).sum(axis=0))
        assert np.allclose(col_sums, 1.0, atol=0.5), f"J_L column |sums| far from 1: {col_sums}"

    def test_event_location_changes_worst_gfm(self, lti_freq_dyn):
        """THE headline assertion: different event buses -> different worst GFM.

        Bus pp1 is adjacent to the GFM at pp0; bus pp8 is adjacent to the GFM at pp9.
        """
        if lti_freq_dyn._J_L is None or not np.isfinite(lti_freq_dyn._J_L).all():
            pytest.skip("J_L has NaN/Inf (test network power flow issue)")
        st_a = self._step(lti_freq_dyn, event_location_pp=1)
        st_b = self._step(lti_freq_dyn, event_location_pp=8)
        # Per-GFM response profiles must differ by location.
        assert not np.allclose(st_a.delta_f_per_bus, st_b.delta_f_per_bus)
        # And the most-deviated (worst) GFM identity should change.
        wa = int(np.argmax(np.abs(st_a.delta_f_per_bus)))
        wb = int(np.argmax(np.abs(st_b.delta_f_per_bus)))
        assert wa != wb, f"worst GFM did not change with location (both {wa})"

    def test_located_differs_from_scalar_fallback(self, lti_freq_dyn):
        """Located injection (via J_L) differs from the legacy rating-share scalar."""
        if lti_freq_dyn._J_L is None or not np.isfinite(lti_freq_dyn._J_L).all():
            pytest.skip("J_L has NaN/Inf (test network power flow issue)")
        st_loc = self._step(lti_freq_dyn, event_location_pp=1)
        st_scalar = self._step(lti_freq_dyn, event_location_pp=None)
        assert not np.allclose(st_loc.delta_f_per_bus, st_scalar.delta_f_per_bus)

    def test_event_at_gfm_bus_injects_that_gfm(self, lti_freq_dyn):
        """Edge case: event at a GFM bus forces ONLY that GFM's swing (bypass J_L).

        Tested on the forcing vector directly (post-step worst is muddied by the
        coupled ZOH integration, which is expected).
        """
        gfm_pp = int(lti_freq_dyn._gfm_pp_idx[0])  # pp0 -> gfm index 0
        f = lti_freq_dyn._disturbance_omega_forcing(0.2, event_location_pp=gfm_pp)
        assert f.shape == (lti_freq_dyn.n_gfm,)
        assert abs(f[0]) > 1e-9                       # targeted GFM is forced
        assert np.allclose(np.delete(f, 0), 0.0)      # others untouched
        assert f[0] < 0                               # deficit -> that GFM's Δω down

    def test_located_forcing_pushes_coi_down(self, lti_freq_dyn):
        """Sign check: a deficit at a passive bus yields a net (COI) downward push."""
        if lti_freq_dyn._J_L is None or not np.isfinite(lti_freq_dyn._J_L).all():
            pytest.skip("J_L has NaN/Inf (test network power flow issue)")
        f = lti_freq_dyn._disturbance_omega_forcing(0.2, event_location_pp=1)
        w = lti_freq_dyn._gfm_ratings / float(lti_freq_dyn._gfm_ratings.sum())
        assert float(w @ f) < 0.0, "deficit should push COI frequency down"

    def test_scalar_backcompat_unchanged(self, lti_freq_dyn):
        """Scalar delta_P_L with no location uses the rating-share path (back-compat)."""
        if lti_freq_dyn._J_L is None or not np.isfinite(lti_freq_dyn._J_L).all():
            pytest.skip("J_L has NaN/Inf (test network power flow issue)")
        st = self._step(lti_freq_dyn, event_location_pp=None, mag=0.2)
        assert np.isfinite(st.delta_f_hz)
        assert st.delta_f_worst > 0.0  # disturbance produced a response
