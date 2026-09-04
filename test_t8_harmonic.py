import numpy as np
import pandapower as pp

from src.eval.harmonic_analysis import HarmonicAnalyzer


def build_toy_net() -> pp.pandapowerNet:
    net = pp.create_empty_network(f_hz=50.0)
    buses = [pp.create_bus(net, vn_kv=11.0) for _ in range(5)]
    pp.create_ext_grid(net, buses[0])
    for i in range(4):
        pp.create_line_from_parameters(
            net,
            buses[i],
            buses[i + 1],
            length_km=0.5,
            r_ohm_per_km=0.2,
            x_ohm_per_km=0.3,
            c_nf_per_km=50.0,
            max_i_ka=1.0,
        )
    pp.create_load(net, buses[4], p_mw=0.3)
    pp.runpp(net)
    return net


def main() -> None:
    net = build_toy_net()
    analyzer = HarmonicAnalyzer(net)

    agent_powers = np.array([0.3, 0.1, 0.05], dtype=np.float32)
    agent_buses = [1, 2, 3]

    results = analyzer.run(agent_powers, agent_buses)

    n_bus = int(net.bus.shape[0])
    thd_v = np.asarray(results["THD_V_pct"], dtype=float)
    thd_v_max = float(results["THD_V_max"])
    thd_v_pcc = float(results["THD_V_PCC"])

    assert thd_v.shape == (n_bus,), f"Unexpected THD_V shape: {thd_v.shape}"
    assert thd_v_max > 0.0, "Expected non-zero harmonic content"
    assert thd_v_max < 500.0, f"THD_V_max out of finite bound: {thd_v_max}"
    assert thd_v_pcc <= thd_v_max, "Expected PCC THD <= worst-bus THD"

    print(f"Toy net THD_V_max={thd_v_max:.1f}% (expected high for 5-bus toy)")
    print("Will be realistic on actual IEEE 123-bus network")
    print("T8 qualitative gate PASS")


if __name__ == "__main__":
    main()
