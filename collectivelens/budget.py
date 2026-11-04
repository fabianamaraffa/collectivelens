"""A daily cap on calls made with the server's own API key.

If you put your key on a public demo, anyone could use it. This counter limits the damage: after
the cap, visitors still get search and passages, and can paste their own key to get written answers.
Calls made with a visitor's own key are never counted.
"""

from __future__ import annotations

import os
import threading
from datetime import date


class DailyBudget:
    def __init__(self, limit: int | None = None) -> None:
        if limit is None:
            raw = os.getenv("COLLECTIVELENS_DAILY_LIMIT", "40")
            limit = int(raw) if raw.strip().lstrip("-").isdigit() else 40
        self.limit = limit
        self._day = date.today()
        self._used = 0
        self._lock = threading.Lock()

    def take(self) -> bool:
        """Use one call if any are left today. A limit of 0 or less means no server calls at all."""
        with self._lock:
            today = date.today()
            if today != self._day:
                self._day, self._used = today, 0
            if self._used >= self.limit:
                return False
            self._used += 1
            return True

    @property
    def remaining(self) -> int:
        with self._lock:
            return max(0, self.limit - (self._used if date.today() == self._day else 0))
