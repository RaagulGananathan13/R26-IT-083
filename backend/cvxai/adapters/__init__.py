"""Thin adapters over the four research components."""

from cvxai.adapters.base import ComponentAdapter
from cvxai.adapters.cxr import CxrAdapter
from cvxai.adapters.ecg import EcgAdapter
from cvxai.adapters.echo import EchoAdapter
from cvxai.adapters.triage import TriageAdapter

#: Registration order is the clinical pathway order, and it is the order the
#: components appear in /health and /components.
ADAPTER_CLASSES = (CxrAdapter, EcgAdapter, EchoAdapter, TriageAdapter)

__all__ = [
    "ADAPTER_CLASSES",
    "ComponentAdapter",
    "CxrAdapter",
    "EcgAdapter",
    "EchoAdapter",
    "TriageAdapter",
]
