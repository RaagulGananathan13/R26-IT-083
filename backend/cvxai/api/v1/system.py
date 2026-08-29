"""Service-level endpoints: health, component registry, component detail."""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends

from cvxai import __project_id__, __version__
from cvxai.api.deps import registry
from cvxai.core.registry import ComponentRegistry
from cvxai.schemas.common import ComponentInfo, HealthReport

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthReport, summary="Service and component health")
def health(reg: ComponentRegistry = Depends(registry)) -> HealthReport:
    """Per-component readiness.

    `status: ok` means at least one component can serve. A component reporting
    `unavailable` carries the reason in `detail` -- usually a missing checkpoint
    or a component root that moved.
    """
    return reg.health(__version__, __project_id__)


@router.get("/components", response_model=List[ComponentInfo],
            summary="Registered components and their model cards")
def components(reg: ComponentRegistry = Depends(registry)) -> List[ComponentInfo]:
    return [reg.describe(cid) for cid in reg.ids()]


@router.get("/components/{component_id}", response_model=ComponentInfo,
            summary="One component, with its published metrics and limitations")
def component_detail(component_id: str,
                     reg: ComponentRegistry = Depends(registry)) -> ComponentInfo:
    return reg.describe(component_id)


@router.get("/cohorts", summary="Dataset provenance and measured cohort overlap")
def cohorts() -> dict:
    """Why the multi-modal endpoint aggregates instead of fusing.

    The claim "no cohort carries all four modalities for the same patient" is
    load-bearing, so it is measured rather than asserted. Regenerate with
    `python scripts/measure_cohort_overlap.py`; the endpoint falls back to the
    documented position when the measurement has not been run on this install.
    """
    from cvxai.settings import get_settings

    path = get_settings().cache_dir / "cohort_overlap.json"
    if path.exists():
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["source"] = "measured on this install"
        return payload
    return {
        "source": "not measured on this install",
        "conclusion": (
            "Components 02 (PTB-XL, Germany, 1989-96) and 03 (EchoNet-Dynamic, "
            "Stanford; CAMUS, France) share no patient identifier with each other "
            "or with the MIMIC-derived Components 01 and 04, so a four-modality "
            "cohort cannot be constructed. Run "
            "`python scripts/measure_cohort_overlap.py` for the exact figures."),
    }


@router.post("/components/{component_id}/warm", response_model=ComponentInfo,
             summary="Load a component's weights ahead of first use")
def warm(component_id: str,
         reg: ComponentRegistry = Depends(registry)) -> ComponentInfo:
    """Pay the load cost now instead of on the first clinical request.

    Useful immediately before a demonstration: Component 01 alone takes about
    thirty seconds to bring two networks into memory.
    """
    reg.get(component_id).warm()
    return reg.describe(component_id)
