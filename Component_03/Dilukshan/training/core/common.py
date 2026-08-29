"""Seeding, device, checkpoint and logging helpers."""
from __future__ import annotations
from pathlib import Path
import os, random, json, csv
import numpy as np
import torch


def seed_everything(seed: int = 1337):
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    # Required by deterministic CUDA matrix multiplies on CUDA >= 10.2.  It
    # must be set before the first CUDA operation in the process.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Reproducibility is more important than the small autotuner speedup for a
    # medical benchmark. warn_only retains portability if a third-party op has
    # no deterministic implementation while making that limitation visible.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def seed_worker(worker_id: int):
    """Seed Python and NumPy from PyTorch's deterministic worker seed."""
    del worker_id  # torch.initial_seed() already incorporates the worker id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_torch_generator(seed: int) -> torch.Generator:
    """Create an explicitly seeded DataLoader/sampler generator."""
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def get_device(pref: str = "cuda") -> torch.device:
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(path: Path, model, optimizer=None, scaler=None,
                    epoch: int = 0, best_metric: float = -1e9, extra: dict = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = dict(model=model.state_dict(), epoch=epoch, best_metric=best_metric,
                extra=extra or {})
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, model, optimizer=None, scaler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    # Historical releases store a wrapper dict; accepting a raw state_dict is
    # inexpensive and makes externally exported checkpoints usable too.
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt


def write_json_atomic(path: Path, payload: dict, *, indent: int = 2):
    """Write JSON without leaving a truncated report after interruption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


class CSVLogger:
    """Append-only CSV logger for per-epoch metrics."""
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._header_written = self.path.exists()

    def log(self, row: dict):
        write_header = not self.path.exists()
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)


def count_params(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def capture_rng_state() -> dict:
    """Snapshot python/numpy/torch RNG state for reproducible resume."""
    state = dict(python=random.getstate(), numpy=np.random.get_state(),
                 torch=torch.get_rng_state())
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _as_cpu_byte(t):
    """RNG states must be CPU uint8 tensors; torch.load(map_location='cuda')
    can move them to the GPU, so coerce back."""
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().to(torch.uint8)
    return t


def restore_rng_state(state: dict):
    if not state:
        return
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(_as_cpu_byte(state["torch"]))
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([_as_cpu_byte(s) for s in state["cuda"]])
    except Exception as e:                              # pragma: no cover
        print(f"[resume][WARN] could not restore RNG state: {e}")
