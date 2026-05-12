"""Mini Optimizer — candidate implementation.

Fill in the `optimize` function below. See README.md for the full spec.

You may add helper functions / classes / modules as needed, but the public
entry point `optimize(...)` must keep the exact signature and return shape
described in the README.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _compute_sharpe(adjusted_pnl: np.ndarray) -> float:
    if adjusted_pnl.size == 0:
        return 0.0

    std = float(np.std(adjusted_pnl, ddof=0))
    if std == 0.0:
        return 0.0

    return float(np.mean(adjusted_pnl)) / std


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
    if trades_df.empty or not stop_losses or not take_profits:
        return []

    df = trades_df[["pnl", "mae", "mfe"]].dropna()
    if df.empty:
        return []

    pnl = df["pnl"].to_numpy()
    mae = df["mae"].to_numpy()
    mfe = df["mfe"].to_numpy()

    results: list[dict] = []

    for sl in stop_losses:
        for tp in take_profits:
            adjusted_pnl = np.where(
                mae >= sl,
                -sl,
                np.where(
                    mfe >= tp,
                    tp,
                    pnl,
                )
            )
            results.append(
                {
                    "stop_loss": float(sl),
                    "take_profit": float(tp),
                    "sharpe": _compute_sharpe(adjusted_pnl),
                    "total_pnl": float(np.sum(adjusted_pnl)),
                    "stopped_out": int(np.count_nonzero(mae >= sl)),
                    "took_profit": int(np.count_nonzero((mae < sl) & (mfe >= tp))),
                }
            )

    results.sort(
        key=lambda x: (-x["sharpe"], -x["total_pnl"], x["stop_loss"], x["take_profit"])
    )

    return results[:top_n]
