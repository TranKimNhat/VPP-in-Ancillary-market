from __future__ import annotations

from warnings import warn

from src.layer0_dso.zonal_pricing import PRICING_METHODS, generate_market_signals

warn(
    "src.layer0_dso.zonalpricing is deprecated; import from src.layer0_dso.zonal_pricing instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["PRICING_METHODS", "generate_market_signals"]
