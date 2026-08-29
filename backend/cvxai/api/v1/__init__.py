"""Version 1 of the API."""

from fastapi import APIRouter

from cvxai.api.v1 import assessment, pathway, studies, system

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(system.router)
api_router.include_router(studies.router)
api_router.include_router(assessment.router)
api_router.include_router(pathway.router)

__all__ = ["api_router"]
