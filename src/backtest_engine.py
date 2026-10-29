"""Reusable, leakage-safe backtesting helpers for next-day predictions."""

import numpy as np


def run_backtest(dates, prices, predictions, actual_next_prices=None):
    """Evaluate predictions where each prediction is for the next price.

    ``prices`` contains the price available when the signal is generated and
    ``predictions`` contains the corresponding next-price predictions.
    ``actual_next_prices`` is used when the next prices are not simply
    ``prices[1:]``, such as when a model's test windows start mid-series.
    """
    del dates

    price_now = np.asarray(prices, dtype=float).reshape(-1)
    pred_next = np.asarray(predictions, dtype=float).reshape(-1)
    if actual_next_prices is None:
        actual_next = price_now[1:]
        price_now = price_now[:-1]
        pred_next = pred_next[:-1]
    else:
        actual_next = np.asarray(actual_next_prices, dtype=float).reshape(-1)

    n = min(len(price_now), len(pred_next), len(actual_next))
    if n < 2:
        return {"error": "not enough data for backtest"}

    price_now = price_now[:n]
    pred_next = pred_next[:n]
    actual_next = actual_next[:n]
    valid = (
        np.isfinite(price_now)
        & np.isfinite(pred_next)
        & np.isfinite(actual_next)
        & (price_now != 0)
    )
    price_now = price_now[valid]
    pred_next = pred_next[valid]
    actual_next = actual_next[valid]
    if len(price_now) == 0:
        return {"error": "no valid observations for backtest"}

    true_returns = (actual_next - price_now) / price_now
    positions = np.where(pred_next > price_now, 1.0, -1.0)
    strategy_returns = positions * true_returns
    daily_std = strategy_returns.std(ddof=1) if len(strategy_returns) > 1 else 0.0
    sharpe = (
        strategy_returns.mean() / daily_std * np.sqrt(252)
        if daily_std > 0
        else 0.0
    )

    return {
        "cumulative_return": float(np.prod(1 + strategy_returns) - 1),
        "daily_returns": strategy_returns.tolist(),
        "sharpe": float(sharpe),
        "directional_accuracy": float(
            np.mean(np.sign(pred_next - price_now) == np.sign(actual_next - price_now))
        ),
        "n_trades": int(len(strategy_returns)),
    }
