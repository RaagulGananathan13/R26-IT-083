"""
The component registry.

Owns adapter construction, lazy loading and health reporting. One instance
lives on `app.state` for the lifetime of the process.
"""
from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional

from cvxai.adapters import ADAPTER_CLASSES, ComponentAdapter
from cvxai.core.errors import ComponentNotFound
from cvxai.core.logging import get_logger
from cvxai.schemas.common import ComponentInfo, ComponentStatus, HealthReport
from cvxai.settings import Settings

log = get_logger("cvxai.registry")


class ComponentRegistry:
    """Holds the four adapters and answers questions about them."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.started_at = time.time()
        self._lock = threading.Lock()
        roots = settings.component_roots()
        self._adapters: Dict[str, ComponentAdapter] = {}
        for adapter_class in ADAPTER_CLASSES:
            adapter = adapter_class(settings, roots.get(adapter_class.id))
            self._adapters[adapter.id] = adapter

    # ------------------------------------------------------------------
    def ids(self) -> List[str]:
        return list(self._adapters)

    def get(self, component_id: str) -> ComponentAdapter:
        adapter = self._adapters.get(component_id)
        if adapter is None:
            raise ComponentNotFound(
                "No component %r. Registered: %s"
                % (component_id, ", ".join(self.ids())),
                {"registered": self.ids()})
        return adapter

    def all(self) -> List[ComponentAdapter]:
        return list(self._adapters.values())

    # ------------------------------------------------------------------
    def warm_all(self) -> None:
        """Eagerly load every serviceable component.

        Off by default. Loading all four costs roughly two minutes and several
        gigabytes of RAM, which is the wrong trade for a demo that may only
        exercise one modality; it is the right trade before a live presentation,
        where a first-request stall is worse than a slow start.
        """
        with self._lock:
            for adapter in self._adapters.values():
                if adapter.status is ComponentStatus.AVAILABLE:
                    adapter.warm()

    # ------------------------------------------------------------------
    def describe(self, component_id: str, include_model: bool = True) -> ComponentInfo:
        adapter = self.get(component_id)
        model = None
        if include_model:
            try:
                model = adapter.model_card()
            except Exception:                  # noqa: BLE001 - never break /health
                model = None
        return ComponentInfo(
            id=adapter.id,
            name=adapter.name,
            owner=adapter.owner,
            modality=adapter.modality,
            task=adapter.task,
            dataset=adapter.dataset,
            status=adapter.status,
            endpoint=adapter.endpoint,
            root=str(adapter.root) if adapter.root else None,
            detail=adapter.status_detail,
            notes=adapter.notes(),
            model=model,
        )

    def health(self, version: str, project_id: str) -> HealthReport:
        components = [self.describe(cid, include_model=False) for cid in self.ids()]
        serviceable = [c for c in components
                       if c.status in (ComponentStatus.READY, ComponentStatus.AVAILABLE)]
        return HealthReport(
            service="cvxai",
            version=version,
            project_id=project_id,
            status="ok" if serviceable else "degraded",
            device=self.device(),
            components=components,
            uptime_s=round(time.time() - self.started_at, 1),
        )

    # ------------------------------------------------------------------
    def device(self) -> str:
        """Report the accelerator actually in use, not the one requested."""
        preference = self.settings.device
        try:
            import torch
        except ImportError:
            return "cpu (torch not installed)"
        available = torch.cuda.is_available()
        if preference in ("auto", "cuda") and available:
            return "cuda:%s" % torch.cuda.get_device_name(0)
        if preference == "cuda" and not available:
            return "cpu (cuda requested but unavailable)"
        return "cpu"


_registry: Optional[ComponentRegistry] = None
_registry_lock = threading.Lock()


def get_registry(settings: Optional[Settings] = None) -> ComponentRegistry:
    """Process-wide registry singleton."""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                from cvxai.settings import get_settings
                _registry = ComponentRegistry(settings or get_settings())
    return _registry


def reset_registry() -> None:
    """Drop the cached registry. Used by the test-suite only."""
    global _registry
    _registry = None
