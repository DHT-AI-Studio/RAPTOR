from fastapi import APIRouter
import requests

from app.core.config import settings

router = APIRouter()


@router.get("/health", tags=["Debug"])
def test_keycloak():
    """測試 FastAPI container 是否可以連上 Keycloak"""
    try:
        url = f"{settings.KEYCLOAK_URL}/realms/master"
        resp = requests.get(url, timeout=5)
        return {
            "keycloak_url": settings.KEYCLOAK_URL,
            "status_code": resp.status_code,
            "reachable": True
        }
    except Exception as e:
        return {
            "keycloak_url": settings.KEYCLOAK_URL,
            "reachable": False,
            "error": str(e)
        }
