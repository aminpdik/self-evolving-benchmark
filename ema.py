"""Exponential Moving Average (EMA) tracking utilities."""


from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class EMATracker:
    """Exponential moving average tracker."""
    alpha: float
    ema: Optional[float] = None

    def update(self, x: float) -> float:
        """Update the exponential moving average (EMA) with a new observation.

Args:
    x: New observed value (e.g., a judge score).

Returns:
    The updated EMA value.

How it works:
    - If this is the first update (ema is None), it initializes ema = x.
    - Otherwise it applies: ema = alpha*x + (1-alpha)*ema."""

        if self.ema is None:
            self.ema = x
        else:
            self.ema = self.alpha * x + (1.0 - self.alpha) * self.ema
        return self.ema