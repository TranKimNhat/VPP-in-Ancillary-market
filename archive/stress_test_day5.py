from __future__ import annotations

import numpy as np

from src.env.make_env import make_vec_envs, make_vec_envs_normalized


USE_DUMMY = True


def run_parallel_gate() -> None:
    # Gate criteria:
    # - obs.shape == (4, 30, 22)
    # - no NaNs in obs
    # - diversity > 0.01
    env_kwargs = {
        "placement_path": "artifacts/placement/official_placement.json",
        "mpc_path": "data/grid_IEEE123_complete.m",
        "precomputed_dir": "data/precomputed",
    }

    print("Creating 4 parallel envs...")
    envs = make_vec_envs(n_envs=4, env_kwargs=env_kwargs, seed=42, use_dummy=USE_DUMMY)
    obs = envs.reset()
    assert obs.shape == (4, 30, 22), f"Bad vec obs shape: {obs.shape}"

    print("Running stress test: 4 envs × 96 steps...")

    for step in range(96):
        actions = np.random.uniform(-1, 1, (4, 54)).astype(np.float32)
        obs, rewards, dones, infos = envs.step(actions)

        assert obs.shape == (4, 30, 22), f"step {step}: bad obs shape"
        assert not np.any(np.isnan(obs)), f"step {step}: NaN detected"

    obs_diversity = obs.std(axis=0).mean()
    print(f"Obs diversity (std across envs): {obs_diversity:.4f}")
    assert obs_diversity > 0.01, "Envs have no diversity"

    envs.close()
    print("GATE DAY 5: PASS")


def run_vecnormalize_gate() -> None:
    env_kwargs = {
        "placement_path": "artifacts/placement/official_placement.json",
        "mpc_path": "data/grid_IEEE123_complete.m",
        "precomputed_dir": "data/precomputed",
    }

    print("Creating 4 parallel envs with VecNormalize...")
    venv = make_vec_envs_normalized(
        n_envs=4, env_kwargs=env_kwargs, seed=42, use_dummy=USE_DUMMY
    )
    obs = venv.reset()
    assert obs.shape == (4, 30, 22), f"Bad vec obs shape: {obs.shape}"

    for _ in range(96):
        actions = np.random.uniform(-1, 1, (4, 54)).astype(np.float32)
        obs, rewards, dones, infos = venv.step(actions)
        assert not np.any(np.isnan(obs)), "NaN detected in obs"

    obs_diversity = obs.std(axis=0).mean()
    print(f"Obs diversity (std across envs): {obs_diversity:.4f}")
    assert obs_diversity > 0.01, "Envs have no diversity"

    venv.close()
    print("GATE DAY 5: PASS")


if __name__ == "__main__":
    run_parallel_gate()
    run_vecnormalize_gate()
