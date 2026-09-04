from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch_geometric.nn import GATConv


def _load_agent_registry(
    placement_path: str = "artifacts/placement/official_placement_v3.json",
) -> Tuple[List[int], List[str], Dict[int, List[int]], List[int]]:
    placement = json.loads(Path(placement_path).read_text(encoding="utf-8"))

    evcs = sorted(placement["evcs"], key=lambda e: int(e["id"][1:]))
    dpv = sorted(placement["dpv"], key=lambda p: int(p["id"][2:]))

    evcs_buses = [e["bus"] for e in evcs]
    dpv_buses = [p["bus"] for p in dpv]

    agent_buses = evcs_buses * 3 + dpv_buses
    agent_types = ["EVCS_PV"] * 9 + ["EVCS_BESS"] * 9 + ["EVCS_V2G"] * 9 + ["DPV"] * 14

    vpp_names = ["VPP_1", "VPP_2", "VPP_3"]
    vpp_agent_indices: Dict[int, List[int]] = {}
    for vpp_idx, vpp_name in enumerate(vpp_names):
        evcs_in_vpp = [i for i, e in enumerate(evcs) if e.get("vpp") == vpp_name]
        dpv_in_vpp = [i for i, p in enumerate(dpv) if p.get("vpp") == vpp_name]

        agent_idxs: List[int] = []
        for evcs_idx in evcs_in_vpp:
            agent_idxs += [evcs_idx, evcs_idx + 9, evcs_idx + 18]
        for dpv_idx in dpv_in_vpp:
            agent_idxs.append(27 + dpv_idx)

        vpp_agent_indices[vpp_idx] = sorted(agent_idxs)

    v2g_agent_indices = list(range(18, 27))

    return agent_buses, agent_types, vpp_agent_indices, v2g_agent_indices


AGENT_BUSES, AGENT_TYPES, VPP_AGENT_INDICES, V2G_AGENT_INDICES = _load_agent_registry()

AGENT_TYPE_DIM = {
    "EVCS_PV": 4,
    "EVCS_BESS": 4,
    "EVCS_V2G": 2,
    "DPV": 4,
}


@dataclass
class ActorOutput:
    actions: torch.Tensor
    log_probs: torch.Tensor
    entropy: torch.Tensor


class GATEncoder(nn.Module):
    def __init__(self, in_dim: int = 24, hidden_dim: int = 32, out_dim: int = 64, heads: int = 4) -> None:
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, concat=True, dropout=0.1)
        self.gat2 = GATConv(hidden_dim * heads, out_dim, heads=1, concat=True, dropout=0.1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() > 0:
            assert int(edge_index.max()) < x.shape[0], (
                f"edge_index max {int(edge_index.max())} exceeds n_nodes {x.shape[0]}"
            )
        h = self.gat1(x, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=0.1, training=self.training)
        h = self.gat2(h, edge_index)
        return h


class TypeConditionedActor(nn.Module):
    def __init__(self, obs_dim: int = 24, emb_dim: int = 64) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.obs_dim = obs_dim

        self.heads = nn.ModuleDict(
            {
                "EVCS_PV": self._make_head(emb_dim + obs_dim, AGENT_TYPE_DIM["EVCS_PV"]),
                "EVCS_BESS": self._make_head(emb_dim + obs_dim, AGENT_TYPE_DIM["EVCS_BESS"]),
                "EVCS_V2G": self._make_head(emb_dim + obs_dim, AGENT_TYPE_DIM["EVCS_V2G"]),
                "DPV": self._make_head(emb_dim + obs_dim, AGENT_TYPE_DIM["DPV"]),
            }
        )


    @staticmethod
    def _make_head(in_dim: int, out_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        obs: torch.Tensor,
        agent_types: List[str],
    ) -> ActorOutput:
        actions = torch.zeros((len(agent_types), 2), device=embeddings.device)
        log_probs = torch.zeros((len(agent_types),), device=embeddings.device)
        entropies = torch.zeros((len(agent_types),), device=embeddings.device)

        for i, agent_type in enumerate(agent_types):
            head = self.heads[agent_type]
            logits = head(torch.cat([embeddings[i], obs[i]], dim=-1))
            std = torch.ones_like(logits)
            dist = Normal(logits, std)
            raw_action = dist.rsample()
            action = torch.tanh(raw_action)
            log_prob = dist.log_prob(raw_action).sum(dim=-1)
            log_prob -= torch.sum(torch.log(1 - action.pow(2) + 1e-6))
            entropy = dist.entropy().sum(dim=-1)

            if agent_type == "EVCS_V2G":
                actions[i, 0] = action[0]
            else:
                actions[i, :2] = action[:2]

            log_probs[i] = log_prob
            entropies[i] = entropy

        return ActorOutput(actions=actions, log_probs=log_probs, entropy=entropies)

    @staticmethod
    def get_flat_actions(actions: torch.Tensor, agent_types: List[str]) -> torch.Tensor:
        flat = []
        for i, agent_type in enumerate(agent_types):
            if agent_type == "EVCS_V2G":
                flat.append(actions[i, :1])
            else:
                flat.append(actions[i, :2])
        return torch.cat(flat, dim=-1).unsqueeze(0)


class VPPCritic(nn.Module):
    def __init__(self, emb_dim: int = 64, global_dim: int = 10) -> None:
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(emb_dim + global_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, embeddings: torch.Tensor, global_state: torch.Tensor) -> torch.Tensor:
        values = []
        for vpp_id, agent_indices in VPP_AGENT_INDICES.items():
            vpp_emb = embeddings[agent_indices].mean(dim=0)
            inp = torch.cat([vpp_emb, global_state], dim=-1)
            values.append(self.critic(inp).squeeze(-1))
        return torch.stack(values, dim=0)


def build_edge_index(net, device: str = "cpu") -> torch.Tensor:
    bus_ids = list(net.bus.index)
    bus_id_map = {bus_id: idx for idx, bus_id in enumerate(bus_ids)}

    edges = []
    for _, row in net.line.iterrows():
        from_bus = int(row["from_bus"])
        to_bus = int(row["to_bus"])
        if from_bus not in bus_id_map or to_bus not in bus_id_map:
            raise ValueError(f"Line references unknown bus: {from_bus}->{to_bus}")
        edges.append((bus_id_map[from_bus], bus_id_map[to_bus]))
        edges.append((bus_id_map[to_bus], bus_id_map[from_bus]))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    if edge_index.numel() > 0:
        assert int(edge_index.max()) < len(bus_ids), (
            f"edge_index max {int(edge_index.max())} exceeds n_buses {len(bus_ids)}"
        )
    return edge_index.to(device)
