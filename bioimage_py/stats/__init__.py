"""Block-wise statistics (reductions)."""
from .stats import max, mean, mean_and_std, min, min_and_max, std
from .unique import unique

__all__ = ["max", "min", "mean", "std", "mean_and_std", "min_and_max", "unique"]
