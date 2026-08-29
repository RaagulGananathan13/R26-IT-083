"""
Component 04 — resumable Optuna studies.

Searches here run for tens of minutes.  Losing that work to a Ctrl-C, a reboot
or a crashed trial is avoidable: every study is backed by a SQLite file, so a
rerun picks up exactly where it stopped and only the outstanding trials are
executed.

    python train_stage2.py 24            # resumes automatically
    python train_stage2.py 24 --fresh    # discard history and start over

The store also survives changes to n_trials: raise the number in the config and
the next run tops the study up rather than restarting it.
"""
from __future__ import annotations

import os
import sys
from typing import Tuple

import optuna

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_p for _p in (
    _SRC, os.path.join(_SRC, "core"), os.path.join(_SRC, "data"),
    os.path.join(_SRC, "models"), os.path.join(_SRC, "analysis"),
) if _p not in sys.path]
from config import MODEL_DIR

STORE_DIR = os.path.join(MODEL_DIR, "optuna")
os.makedirs(STORE_DIR, exist_ok=True)


def _storage(db_name: str):
    path = os.path.join(STORE_DIR, f"{db_name}.db").replace("\\", "/")
    # A generous busy timeout keeps concurrent trial workers from tripping over
    # SQLite's write lock when several finish at once.
    return optuna.storages.RDBStorage(
        url=f"sqlite:///{path}",
        engine_kwargs={"connect_args": {"timeout": 60}},
    )


def get_study(name: str, db_name: str, seed: int, n_trials: int,
              fresh: bool = False, n_startup: int = 10
              ) -> Tuple[optuna.Study, int, int]:
    """
    Returns (study, n_remaining, n_already_done).

    n_remaining is what the caller should pass to `study.optimize`.
    """
    storage = _storage(db_name)
    if fresh:
        try:
            optuna.delete_study(study_name=name, storage=storage)
        except Exception:
            pass

    study = optuna.create_study(
        study_name=name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup),
    )
    done = len([t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE])
    return study, max(0, n_trials - done), done


def describe(study: optuna.Study, done: int, remaining: int) -> str:
    if done == 0:
        return "new study"
    msg = f"resumed: {done} trial(s) already complete, {remaining} remaining"
    try:
        msg += f"  (best so far {study.best_value:.4f} @ trial {study.best_trial.number})"
    except Exception:
        pass
    return msg


def wants_fresh(argv=None) -> bool:
    argv = sys.argv if argv is None else argv
    return "--fresh" in argv or os.environ.get("C4_FRESH", "") == "1"
