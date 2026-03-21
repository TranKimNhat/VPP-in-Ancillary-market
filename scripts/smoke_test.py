from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.env.microgrid_env import MicrogridEnv
from src.env.make_env import make_vec_envs
from src.opt.l0_reconfig import L0Optimizer, build_net_data_from_pandapower
from src.opt.l1_dispatch import L1Dispatcher
from src.opt.precompute import generate_all_days
from src.rl.networks import GATEncoder, TypeConditionedActor, VPPCritic, build_edge_index


DATA_DIR = Path("data/precomputed")
PLACEMENT_PATH = "artifacts/placement/official_placement.json"
MPC_PATH = "data/grid_IEEE123_complete.m"


def _timeit(fn):
    start = time.time()
    result = fn()
    return result, time.time() - start


def _record(results, name, status, elapsed, error=None):
    results.append({"name": name, "status": status, "time": elapsed, "error": error})


def _fail(msg):
    raise RuntimeError(msg)


def test_parquet_dataset():
    if not DATA_DIR.exists():
        _fail(f"Missing {DATA_DIR}")

    files = sorted(DATA_DIR.glob("day_*.parquet"))
    if len(files) != 200:
        _fail(f"Expected 200 parquet files, found {len(files)}")

    check_days = [0, 50, 100, 199]
    for day in check_days:
        path = DATA_DIR / f"day_{day:03d}.parquet"
        if not path.exists():
            _fail(f"Missing parquet file {path}")
        df = pd.read_parquet(path)
        if len(df) != 96:
            _fail(f"{path.name} expected 96 rows, got {len(df)}")

        for col in ["p_ref_vpp1", "p_ref_vpp2", "p_ref_vpp3"]:
            if df[col].std() <= 0:
                _fail(f"{path.name} {col} does not vary")

        for col in ["lambda_p2p", "lambda_as_ffr", "lambda_as_pfr", "lambda_as_sfr", "lambda_q"]:
            if (df[col] <= 0).any():
                _fail(f"{path.name} {col} has non-positive values")

        freq_count = int(df["freq_event_flag"].sum())
        if not (60 <= freq_count <= 96):
            _fail(f"{path.name} freq_event_flag count out of range: {freq_count}")


def test_single_env():
    env = MicrogridEnv(
        placement_path=PLACEMENT_PATH,
        mpc_path=MPC_PATH,
        precomputed_dir=str(DATA_DIR),
    )
    obs, info = env.reset()
    if obs.shape != (30, 22):
        _fail(f"obs shape mismatch: {obs.shape}")

    reward_keys = {
        "r_track",
        "r_volt",
        "r_freq",
        "r_as",
        "r_deg",
        "r_oblig",
        "r_p2p",
        "r_s_margin",
        "r_q_revenue",
    }

    for _ in range(96):
        action = np.random.uniform(-1, 1, (54,)).astype(np.float32)
        obs, reward, done, truncated, info = env.step(action)
        if obs.shape != (30, 22):
            _fail(f"obs shape mismatch during step: {obs.shape}")
        if np.isnan(obs).any():
            _fail("NaN in observations")
        if not math.isfinite(float(reward)):
            _fail("Reward not finite")

    breakdown = info.get("reward_breakdown", {})
    if set(breakdown.keys()) != reward_keys:
        _fail(f"reward_breakdown keys mismatch: {sorted(breakdown.keys())}")

    for key in [
        "stage1_activations",
        "stage2_activations",
        "stage3_activations",
        "stage4_activations",
        "stage5_activations",
        "total_safety_activations",
    ]:
        if key not in info:
            _fail(f"Missing safety key in info: {key}")

    env.close()


def test_vecenv_diversity():
    env_kwargs = {
        "placement_path": PLACEMENT_PATH,
        "mpc_path": MPC_PATH,
        "precomputed_dir": str(DATA_DIR),
    }
    envs = make_vec_envs(n_envs=4, env_kwargs=env_kwargs, seed=42, use_dummy=True)
    obs = envs.reset()
    if obs.shape != (4, 30, 22):
        _fail(f"vec obs shape mismatch: {obs.shape}")

    for _ in range(96):
        actions = np.random.uniform(-1, 1, (4, 54)).astype(np.float32)
        obs, rewards, dones, infos = envs.step(actions)
        if np.isnan(obs).any():
            _fail("NaN in vec obs")

    diversity = obs.std(axis=0).mean()
    if diversity <= 0.01:
        _fail(f"Obs diversity too low: {diversity:.4f}")

    envs.close()


def _build_agent_types():
    return [
        "EVCS_PV",
        "EVCS_PV",
        "EVCS_PV",
        "EVCS_BESS",
        "EVCS_BESS",
        "EVCS_BESS",
        "EVCS_V2G",
        "EVCS_V2G",
        "EVCS_V2G",
        "EVCS_V2G",
        "EVCS_V2G",
        "EVCS_V2G",
        "DPV",
        "DPV",
        "DPV",
    ] * 2


def test_networks():
    env = MicrogridEnv(
        placement_path=PLACEMENT_PATH,
        mpc_path=MPC_PATH,
        precomputed_dir=str(DATA_DIR),
    )
    edge_index = build_edge_index(env.net)
    n_nodes = int(env.net.bus.shape[0])
    x_full = torch.randn(n_nodes, 22)

    encoder = GATEncoder()
    actor = TypeConditionedActor()
    critic = VPPCritic()

    embeddings = encoder(x_full, edge_index)
    agent_types = _build_agent_types()
    agent_obs = torch.randn(30, 22)
    global_state = torch.randn(10)

    actor_out = actor(embeddings[:30], agent_obs, agent_types)
    actions = actor_out.actions
    values = critic(embeddings[:30], global_state)

    if actions.shape != (30, 2):
        _fail(f"actions shape mismatch: {actions.shape}")
    if values.shape != (3,):
        _fail(f"values shape mismatch: {values.shape}")

    loss = -actor_out.log_probs.mean() + values.sum()
    loss.backward()
    for name, param in encoder.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            _fail(f"NaN gradient in {name}")

    env.close()


def test_l0_l1_pipeline():
    env = MicrogridEnv(
        placement_path=PLACEMENT_PATH,
        mpc_path=MPC_PATH,
    )
    net_data = build_net_data_from_pandapower(env.base_net)

    print(f"net_data buses: {len(net_data['buses'])}")
    print(f"net_data branches: {len(net_data['branches'])}")
    print(f"net_data ders: {len(net_data['ders'])}")
    der_pmax = [d[1] for d in net_data["ders"]]
    print(f"DER P_max values: {der_pmax[:5]}...")
    total_p_max = sum(der_pmax)
    print(f"Total DER P_max: {total_p_max:.1f} MW")
    for d in net_data["ders"]:
        print(f"  bus={d[0]}, P_max={d[1]:.3f}, Q_max={d[2]:.3f}, type={d[3]}")
    if not any(p > 0 for p in der_pmax):
        _fail("All DERs have P_max=0")
    if total_p_max <= 15.0:
        _fail(f"Total P_max too low: {total_p_max:.1f} MW")

    total_storage = sum(d[1] for d in net_data["ders"] if d[3] == "bess")
    print(f"Total storage P_max: {total_storage:.1f} MW")

    if len(net_data["buses"]) != 123:
        _fail(f"Expected 123 buses, got {len(net_data['buses'])}")
    if len(net_data["branches"]) != 122:
        _fail(f"Expected 122 branches, got {len(net_data['branches'])}")
    if not (24 <= len(net_data["ders"]) <= 36):
        _fail(f"DER count out of expected range: {len(net_data['ders'])}")
    if len(net_data["cap_banks"]) < 4:
        _fail(
            f"Expected ≥4 cap banks, got {len(net_data['cap_banks'])}. "
            f"Check net.shunt: {env.net.shunt}"
        )

    print(f"net.shunt: {len(env.net.shunt)} entries")
    print(
        f"Buses: {len(net_data['buses'])}, "
        f"Branches: {len(net_data['branches'])}, "
        f"DERs: {len(net_data['ders'])}, "
        f"CapBanks: {len(net_data['cap_banks'])}"
    )

    optimizer = L0Optimizer(net_data)
    print(f"L0 n_bus: {optimizer.n_bus}")
    print(f"L0 n_branch: {optimizer.n_branch}")
    vpp_caps = {
        1: {"P_max": 10.3, "Q_max": 3.5, "S_agg": 7.01},
        2: {"P_max": 10.3, "Q_max": 3.5, "S_agg": 7.01},
        3: {"P_max": 4.9, "Q_max": 2.0, "S_agg": 5.17},
    }
    profiles = pd.Series(
        {
            "pv_pu": 0.7,
            "wind_mw": 8.0,
            "load_z1": 5.0,
            "load_z2": 6.0,
            "load_z3": 4.5,
            "load_z4": 3.0,
        }
    )

    l0_result, l0_time = _timeit(lambda: optimizer.solve(1, profiles, vpp_caps))
    if l0_time > 30.0:
        _fail(f"L0 solve too slow: {l0_time:.2f}s")
    if l0_result.status not in {"optimal", "optimal_inaccurate"}:
        _fail(f"L0 status: {l0_result.status}")

    print(f"L0 status: {l0_result.status}, time: {l0_result.solve_time * 1000:.0f}ms")

    dispatcher = L1Dispatcher()
    p_refs = []
    for step in range(10):
        pv = max(0.0, 0.8 * np.sin(np.pi * step / 48.0))
        l1_result = dispatcher.solve_step(
            step,
            {"pv_pu": pv, "lambda_as_ffr": 10.0, "lambda_q": 2.0},
            l0_result,
        )
        if l1_result.status != "optimal":
            _fail(f"L1 step {step} failed: {l1_result.status}")
        p_refs.append(l1_result.p_ref[1])

    if np.std(p_refs) <= 0.01:
        _fail("p_ref must vary with PV")

    print(f"L1 p_ref_vpp1 range: [{min(p_refs):.3f}, {max(p_refs):.3f}]")
    print("L0/L1 GATE: PASS")

    tmp_dir = Path("artifacts") / "smoke_precompute"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    generate_all_days(n_days=1, output_dir=str(tmp_dir), seed=1, use_optimizers=True)
    df = pd.read_parquet(tmp_dir / "day_000.parquet")
    if df["p_ref_vpp1"].std() <= 0.0:
        _fail("Optimizer parquet p_ref_vpp1 is constant")


def main():
    results = []
    failures = 0

    for name, test in [
        ("Parquet dataset", test_parquet_dataset),
        ("Single env", test_single_env),
        ("VecEnv diversity", test_vecenv_diversity),
        ("Networks", test_networks),
        ("L0/L1 optimizer", test_l0_l1_pipeline),
    ]:
        try:
            _, elapsed = _timeit(test)
            _record(results, name, "PASS", elapsed)
        except Exception as exc:
            failures += 1
            _record(results, name, "FAIL", 0.0, str(exc))

    total_time = sum(item["time"] for item in results)

    print("\nComponent          | Status | Time")
    print("-------------------|--------|------")
    for item in results:
        print(f"{item['name']:<19} | {item['status']:<6} | {item['time']:.1f}s")
    print("-------------------|--------|------")
    print(f"WEEK 1 TOTAL       | {'FAIL' if failures else 'PASS':<6} | {total_time:.1f}s")

    for item in results:
        if item["status"] == "FAIL":
            print(f"\n[ERROR] {item['name']}: {item['error']}")

    if failures:
        sys.exit(1)

    print("\nWEEK 1 COMPLETE — Ready for RL training")
    sys.exit(0)


if __name__ == "__main__":
    main()
