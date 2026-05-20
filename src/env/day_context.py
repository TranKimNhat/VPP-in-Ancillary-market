from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


_VALID_SPLITS = ("train", "eval", "all")


class DayContextLoader:
    """Load precomputed daily profiles with a train/eval split.

    The precompute directory is expected to contain ``day_*.parquet`` files
    and (optionally) an ``eval_days.txt`` file listing the day filenames
    reserved for held-out evaluation. When ``split='train'`` the loader serves
    only days NOT in that list; when ``split='eval'`` it serves only those
    listed; when ``split='all'`` it serves every day file.
    """

    def __init__(
        self,
        precomputed_dir: str | Path = "data/precomputed_365d_97to67",
        seed: int | None = None,
        preloaded_days: list[pd.DataFrame] | None = None,
        split: str = "train",
    ) -> None:
        if split not in _VALID_SPLITS:
            raise ValueError(f"split must be one of {_VALID_SPLITS}, got '{split}'")
        self.precomputed_dir = Path(precomputed_dir)
        self.split = split
        self._rng = np.random.default_rng(seed)
        self._all_days = preloaded_days if preloaded_days is not None else self._load_all_days()

    def _read_eval_filenames(self) -> set[str]:
        eval_file = self.precomputed_dir / "eval_days.txt"
        if not eval_file.exists():
            return set()
        try:
            return {line.strip() for line in eval_file.read_text().splitlines() if line.strip()}
        except Exception:
            return set()

    def _load_all_days(self) -> list[pd.DataFrame]:
        if not self.precomputed_dir.exists():
            return []
        parquet_files = sorted(self.precomputed_dir.glob("day_*.parquet"))
        if self.split == "all":
            selected = parquet_files
        else:
            eval_names = self._read_eval_filenames()
            if self.split == "eval":
                selected = [p for p in parquet_files if p.name in eval_names]
            else:  # train
                selected = [p for p in parquet_files if p.name not in eval_names]
        return [pd.read_parquet(path) for path in selected]

    def sample_day(self) -> pd.DataFrame:
        if not self._all_days:
            return pd.DataFrame(
                {
                    "step": np.arange(96, dtype=np.int64),
                    "lambda_p2p": np.full(96, 50.0, dtype=np.float32),
                    "lambda_p2p_z1": np.full(96, 50.0, dtype=np.float32),
                    "lambda_p2p_z2": np.full(96, 50.0, dtype=np.float32),
                    "lambda_p2p_z4": np.full(96, 50.0, dtype=np.float32),
                    "lambda_as_ffr": np.full(96, 10.0, dtype=np.float32),
                }
            )
        idx = int(self._rng.integers(0, len(self._all_days)))
        return self._all_days[idx]
