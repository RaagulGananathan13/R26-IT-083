"""Version 1 of the API."""

from fastapi import APIRouter

from cvxai.api.v1 import assessment, studies, system

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(system.router)
api_router.include_router(studies.router)
api_router.include_router(assessment.router)

__all__ = ["api_router"]
