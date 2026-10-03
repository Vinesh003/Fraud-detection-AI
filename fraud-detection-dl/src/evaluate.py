"""Metrics suited to extreme class imbalance + business-cost threshold selection."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def pr_auc(y, s) -> float:
    return float(average_precision_score(y, s))


def recall_at_precision(y, s, min_precision: float = 0.9) -> float:
    p, r, _ = precision_recall_curve(y, s)
    ok = r[p >= min_precision]
    return float(ok.max()) if len(ok) else 0.0


def total_cost(y, flagged, amount, review_cost: float) -> float:
    """Missed fraud costs its transaction amount; every flagged transaction costs a manual review."""
    missed = ((y == 1) & (~flagged)) * amount
    return float(missed.sum() + review_cost * flagged.sum())


def best_threshold(y, s, amount, review_cost: float = 5.0, n_grid: int = 400):
    """Pick the score threshold that minimises total cost. Fit on VALIDATION data only."""
    grid = np.unique(np.quantile(s, np.linspace(0.0, 1.0, n_grid)))
    costs = [total_cost(y, s >= t, amount, review_cost) for t in grid]
    k = int(np.argmin(costs))
    return float(grid[k]), float(costs[k])


def amount_rule_baseline(y, amount, review_cost: float = 5.0, top_frac: float = 0.01):
    """Naive rule: manually review the top x% largest transactions."""
    flagged = amount >= np.quantile(amount, 1 - top_frac)
    return total_cost(y, flagged, amount, review_cost)


def cost_curve(y, s, amount, review_cost: float, n_grid: int = 200):
    grid = np.unique(np.quantile(s, np.linspace(0.0, 1.0, n_grid)))
    return grid, np.array([total_cost(y, s >= t, amount, review_cost) for t in grid])
