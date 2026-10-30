"""
Backtesting Simulation Engine for Quantitative Trading Strategies
==================================================================
Simulates long/short and long-only equity performance, calculating:
- Cumulative Strategy Return vs Buy & Hold Benchmark
- Annualized Sharpe Ratio (assuming 252 trading days/year)
- Maximum Drawdown (MDD)
- Win Rate (% profitable trades)
- Directional Prediction Accuracy
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Sequence


def run_backtest(
    prices: Sequence[float],
    predictions: Sequence[float],
    dates: Optional[Sequence[Any]] = None,
    initial_capital: float = 10000.0,
    long_only: bool = True,
    commission_bps: float = 0.0005
) -> Dict[str, Any]:
    """
    Executes an out-of-sample trading backtest comparing predicted vs actual price.

    Args:
        prices: 1D array of actual asset close prices.
        predictions: 1D array of next-day predicted close prices.
        dates: Optional sequence of date strings / timestamps.
        initial_capital: Starting portfolio equity in USD ($10,000 default).
        long_only: If True, positions are 1 (long) or 0 (cash). If False, +1 (long) or -1 (short).
        commission_bps: Transaction cost per trade (0.05% default).

    Returns:
        dict containing summary statistics, equity curve, and daily returns.
    """
    actual = np.asarray(prices, dtype=float).flatten()
    preds = np.asarray(predictions, dtype=float).flatten()

    n = min(len(actual), len(preds))
    if n < 3:
        return {'error': 'Insufficient data points for backtesting (minimum 3 required)'}

    actual = actual[:n]
    preds = preds[:n]

    # Generate signals: 
    # If predicted next-day price > current actual price -> enter Long (1.0)
    # Otherwise -> 0.0 (cash/flat in long_only) or -1.0 (short)
    price_now = actual[:-1]
    price_next = actual[1:]
    pred_next = preds[1:]

    actual_pct_change = (price_next - price_now) / price_now

    if long_only:
        positions = np.where(pred_next > price_now, 1.0, 0.0)
    else:
        positions = np.where(pred_next > price_now, 1.0, -1.0)

    # Strategy returns minus transaction slippage/commission on position changes
    position_changes = np.abs(np.diff(np.insert(positions, 0, 0.0)))
    costs = position_changes * commission_bps
    daily_strategy_returns = (positions * actual_pct_change) - costs

    # Cumulative equity curve
    equity_curve = initial_capital * np.cumprod(1 + daily_strategy_returns)
    buy_hold_equity = initial_capital * np.cumprod(1 + actual_pct_change)

    total_strategy_return = float((equity_curve[-1] - initial_capital) / initial_capital)
    total_buy_hold_return = float((buy_hold_equity[-1] - initial_capital) / initial_capital)

    # Sharpe ratio
    daily_mean = np.mean(daily_strategy_returns)
    daily_std = np.std(daily_strategy_returns)
    sharpe = float((daily_mean / daily_std) * np.sqrt(252)) if daily_std > 0 else 0.0

    # Max Drawdown
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max
    max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Directional Accuracy
    pred_dir = np.diff(preds) > 0
    actual_dir = np.diff(actual) > 0
    dir_acc = float(np.mean(pred_dir == actual_dir) * 100)

    # Win Rate
    active_trades = daily_strategy_returns[positions != 0]
    win_rate = float(np.mean(active_trades > 0) * 100) if len(active_trades) > 0 else 0.0

    date_strings = [str(d) for d in dates[:n]] if dates is not None else [f"Day_{i}" for i in range(n)]

    return {
        'initial_capital': initial_capital,
        'final_strategy_equity': float(equity_curve[-1]),
        'final_buy_hold_equity': float(buy_hold_equity[-1]),
        'strategy_return_pct': total_strategy_return * 100,
        'buy_hold_return_pct': total_buy_hold_return * 100,
        'sharpe_ratio': round(sharpe, 3),
        'max_drawdown_pct': round(max_drawdown * 100, 2),
        'directional_accuracy_pct': round(dir_acc, 2),
        'win_rate_pct': round(win_rate, 2),
        'total_trading_days': int(n - 1),
        'dates': date_strings[1:],
        'strategy_equity_curve': equity_curve.tolist(),
        'buy_hold_equity_curve': buy_hold_equity.tolist(),
        'daily_strategy_returns': daily_strategy_returns.tolist()
    }
