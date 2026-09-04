from src.env.microgrid_env import MicrogridEnv
import numpy as np

env = MicrogridEnv(
    placement_path="artifacts/placement/official_placement.json",
    mpc_path="data/grid_IEEE123_complete.m",
)

obs, info = env.reset(seed=42)
print(f"obs shape: {obs.shape}")
assert obs.shape == (30, 24)

rewards = []
safety_acts = []

for t in range(96):
    action = np.zeros(54, dtype=np.float32)
    obs, reward, done, truncated, info = env.step(action)

    assert obs.shape == (30, 24), f"t={t}: bad obs shape"
    assert not np.any(np.isnan(obs)), f"t={t}: NaN in obs"
    assert np.isfinite(reward), f"t={t}: non-finite reward"
    assert -10.0 <= reward <= 2.0, f"t={t}: reward out of bounds: {reward}"

    rewards.append(reward)
    safety_acts.append(info.get("total_safety_activations", 0))

    if t % 24 == 0:
        print(
            f"t={t:3d}: r={reward:.4f}, v_min={info['v_min']:.4f}, "
            f"safety={info.get('total_safety_activations', 0)}"
        )

print(f"reward range: [{min(rewards):.4f}, {max(rewards):.4f}]")
print(f"total safety activations: {sum(safety_acts)}")
print("GATE DAY 3: PASS")
