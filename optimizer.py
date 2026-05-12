"""Mini Optimizer — candidate implementation.

Fill in the `optimize` function below. See README.md for the full spec.

You may add helper functions / classes / modules as needed, but the public
entry point `optimize(...)` must keep the exact signature and return shape
described in the README.
"""

from __future__ import annotations

import pandas as pd


def optimize(
    trades_df: pd.DataFrame,
    stop_losses: list[float],
    take_profits: list[float],
    top_n: int = 5,
) -> list[dict]:
    """Return the top-N (stop_loss, take_profit) combinations by Sharpe.

    See README.md for:
      - input data schema (trade_id, entry_time, exit_time, pnl, mae, mfe)
      - SL/TP application rules (SL takes priority over TP)
      - Sharpe definition (mean / std, sharpe = 0.0 when degenerate)
      - required result dict keys
      - tie-breaking and deterministic ordering rules
      - edge cases you must handle
    """
    raise NotImplementedError("Implement me.")
