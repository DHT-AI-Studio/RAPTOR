from fastapi import APIRouter
import requests
import os

router = APIRouter()

KEYCLOAK_URL = os.getenv("KEYCLOAK_URL", "http://keycloak:8080")


@router.get("/keycloak-health", tags=["Debug"])
def test_keycloak():
    """
    測試 FastAPI container 是否可以連上 Keycloak
    """
    try:
        url = f"{KEYCLOAK_URL}/realms/master"
        resp = requests.get(url, timeout=5)

        return {
            "keycloak_url": KEYCLOAK_URL,
            "status_code": resp.status_code,
            "reachable": True
        }

    except Exception as e:
        return {
            "keycloak_url": KEYCLOAK_URL,
            "reachable": False,
            "error": str(e)
        }
