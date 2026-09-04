"""Expand actor.mean_head action_dim 1 -> 2 in a legacy checkpoint.

Justification:
 - The current GAMAPPOAgent expects action_dim_per_agent = 2, but the saved
   am_mappo_ep*.pt checkpoints in artifacts/checkpoints_am_mappo_proposed/
   were trained with action_dim = 1.
 - mean_head is zero-initialized at training start
   (src/rl/train_am_mappo.py:424-425), so padding the new dim 1 row with
   zeros is exactly equivalent to "started training but never updated".
 - log_std is initialized to a constant (line 429); we repeat the trained
   value for the new dim.
 - In the eval path (eval_ffr_topology.py:333) only `[ai, 0]` is consumed
   for the VPP action, so dim 1 outputs are silent.

This is a workaround, not a replacement for re-training.

Usage:
    python scripts/patch_checkpoint_action_dim.py <input.pt> <output.pt>
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch


def patch(src: Path, dst: Path, new_dim: int = 2) -> None:
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    sd = ckpt["agent_state_dict"]

    w = sd["actor.mean_head.weight"]   # [1, H]
    b = sd["actor.mean_head.bias"]     # [1]
    ls = sd["actor.log_std"]           # [1]

    if w.shape[0] == new_dim:
        print(f"Already action_dim={new_dim}, copying as-is.")
        torch.save(ckpt, dst)
        return

    pad_rows = new_dim - w.shape[0]
    new_w = torch.cat([w, torch.zeros(pad_rows, w.shape[1], dtype=w.dtype)], dim=0)
    new_b = torch.cat([b, torch.zeros(pad_rows, dtype=b.dtype)], dim=0)
    new_ls = torch.cat([ls, ls[-1:].repeat(pad_rows)], dim=0)

    sd["actor.mean_head.weight"] = new_w
    sd["actor.mean_head.bias"] = new_b
    sd["actor.log_std"] = new_ls

    torch.save(ckpt, dst)
    print(f"Patched {src.name}: mean_head {tuple(w.shape)} -> {tuple(new_w.shape)} "
          f"-> {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    patch(Path(sys.argv[1]), Path(sys.argv[2]))
