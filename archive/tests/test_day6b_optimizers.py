from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.opt.precompute import generate_all_days


def test_day6b_precompute_outputs(tmp_path: Path) -> None:
    out_dir = tmp_path / "precomputed"
    generate_all_days(n_days=1, output_dir=str(out_dir), seed=1, use_optimizers=True)

    parquet_path = out_dir / "day_000.parquet"
    assert parquet_path.exists()

    df = pd.read_parquet(parquet_path)
    required = [
        "p_ref_vpp1",
        "p_ref_vpp2",
        "p_ref_vpp3",
        "r_as_vpp1",
        "r_as_vpp2",
        "r_as_vpp3",
        "q_commit_vpp1",
        "q_commit_vpp2",
        "q_commit_vpp3",
        "lambda_p2p",
        "lambda_as_ffr",
        "lambda_q",
    ]
    for col in required:
        assert col in df.columns
        assert bool(df[col].notna().all())
