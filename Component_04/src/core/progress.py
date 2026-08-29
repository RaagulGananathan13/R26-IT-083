"""
Component 04 — live progress reporting for long Optuna searches.

Optuna's built-in bar vanishes when output is piped and says nothing about
which trial is winning.  This prints a self-refreshing line carrying elapsed
time, ETA, throughput, the current best value, and live GPU memory, and it
redraws on a timer so the display stays alive even while a single trial is
still running (with parallel trials the first callback can be a minute away).
"""
from __future__ import annotations

import sys
import threading
import time
from typing import Optional

# ---------------------------------------------------------------------------
# GPU probe — reports whole-device usage, so it matches Task Manager.
# ---------------------------------------------------------------------------
_GPU_OK = None


def gpu_mem() -> Optional[tuple]:
    """(used_GB, total_GB) for device 0, or None when unavailable."""
    global _GPU_OK
    if _GPU_OK is False:
        return None
    try:
        import torch
        if not torch.cuda.is_available():
            _GPU_OK = False
            return None
        free, total = torch.cuda.mem_get_info(0)
        _GPU_OK = True
        return ((total - free) / 1e9, total / 1e9)
    except Exception:
        _GPU_OK = False
        return None


def _fmt(sec: float) -> str:
    sec = max(0, int(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class _Base:
    """Shared redraw machinery with a background ticker."""

    def __init__(self, total: int, label: str, width: int, enabled: bool,
                 tick: float = 2.0):
        self.total = max(int(total), 1)
        self.label = label
        self.width = width
        self.enabled = enabled
        self.done = 0
        self.t0 = time.time()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._extra = ""
        self._thread: Optional[threading.Thread] = None
        if self.enabled and tick > 0:
            self._thread = threading.Thread(target=self._ticker, args=(tick,),
                                            daemon=True)
            self._thread.start()

    # ------------------------------------------------------------------
    def _ticker(self, tick: float) -> None:
        while not self._stop.wait(tick):
            with self._lock:
                self._draw()

    def _bar(self, frac: float) -> str:
        filled = int(round(frac * self.width))
        return "#" * filled + "." * (self.width - filled)

    def _draw(self, newline: bool = False) -> None:
        frac = self.done / self.total
        el = time.time() - self.t0
        eta = (el / self.done) * (self.total - self.done) if self.done else 0.0
        rate = self.done / el * 60 if el > 0 else 0.0
        msg = (f"\r  {self.label:<9} [{self._bar(frac)}] {self.done:>3}/{self.total} "
               f"{frac*100:5.1f}%  {_fmt(el)} elapsed")
        if self.done:
            msg += f"  eta {_fmt(eta)}  {rate:4.1f}/min"
        else:
            msg += "  eta --:--  (first trial in flight)"
        if self._extra:
            msg += f"  {self._extra}"
        g = gpu_mem()
        if g:
            msg += f"  gpu {g[0]:.1f}/{g[1]:.1f}GB"
        sys.stdout.write(msg + "  ")
        if newline:
            sys.stdout.write("\n")
        sys.stdout.flush()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        if self.enabled:
            with self._lock:
                self._draw(newline=True)


class TrialProgress(_Base):
    """Optuna callback: `study.optimize(..., callbacks=[TrialProgress(n, 'xgb')])`."""

    def __init__(self, total: int, label: str = "", width: int = 24,
                 enabled: bool = True, tick: float = 2.0):
        super().__init__(total, label, width, enabled, tick)

    def __call__(self, study, trial) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.done += 1
            try:
                self._extra = (f"last={trial.value:.4f} "
                               f"best={study.best_value:.4f}@{study.best_trial.number}")
            except Exception:
                self._extra = ""
            self._draw(newline=self.done >= self.total)


class StepProgress(_Base):
    """Plain counter for non-Optuna loops (CV folds, bootstrap fits)."""

    def __init__(self, total: int, label: str = "", width: int = 24,
                 enabled: bool = True, tick: float = 2.0):
        super().__init__(total, label, width, enabled, tick)

    def step(self, note: str = "") -> None:
        if not self.enabled:
            return
        with self._lock:
            self.done += 1
            self._extra = note
            self._draw(newline=self.done >= self.total)
