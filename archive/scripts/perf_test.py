from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from src.env.microgrid_env import MicrogridEnv
from src.rl.networks import AGENT_TYPES, GATEncoder, TypeConditionedActor, build_edge_index


DATA_DIR = Path("data/precomputed_365d_97to67")
PLACEMENT_PATH = "artifacts/placement/official_placement_v3.json"
MPC_PATH = "data/grid_IEEE123_complete.m"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = MicrogridEnv(
        placement_path=PLACEMENT_PATH,
        mpc_path=MPC_PATH,
        precomputed_dir=str(DATA_DIR),
    )
    obs, _ = env.reset()
    edge_index = build_edge_index(env.net)
    bus_ids = list(env.net.bus.index)
    bus_id_map = {bus_id: idx for idx, bus_id in enumerate(bus_ids)}
    agent_bus_indices = [bus_id_map[env.agent_bus_map[i][0]] for i in range(len(AGENT_TYPES))]

    encoder = GATEncoder().to(device)
    actor = TypeConditionedActor().to(device)

    print(torch.cuda.is_available())
    print(next(actor.parameters()).device)

    t0 = time.time()
    for _ in range(100):
        obs, r, done, _, info = env.step(env.action_space.sample())
    print(f"env.step ×100: {time.time() - t0:.2f}s")

    x_full = torch.zeros((len(bus_ids), obs.shape[1]), dtype=torch.float32, device=device)
    agent_obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    x_full[torch.tensor(agent_bus_indices, device=device)] = agent_obs
    edge_index_t = torch.as_tensor(edge_index, dtype=torch.long, device=device)

    t0 = time.time()
    with torch.no_grad():
        for _ in range(100):
            embeddings = encoder(x_full, edge_index_t)
            agent_embeddings = embeddings[agent_bus_indices]
            _ = actor(agent_embeddings, agent_obs, AGENT_TYPES)
    print(f"actor forward ×100: {time.time() - t0:.2f}s")

    env.close()


if __name__ == "__main__":
    main()
