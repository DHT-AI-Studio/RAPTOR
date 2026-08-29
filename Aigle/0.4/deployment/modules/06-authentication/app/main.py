from contextlib import asynccontextmanager
import asyncio
import logging

from fastapi import FastAPI, Depends
from fastapi.responses import RedirectResponse

from app.core.security import APIsecurity
from app.routers import account, auth, creat_user, group, health, login, path
from app import sync_resources

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await asyncio.to_thread(
            sync_resources.sync,
            apply=True, verbose=False, retries=6, retry_delay=10
        )
    except Exception as e:
        logger.warning(f"Keycloak resource sync failed (non-fatal): {e}")
    yield


app = FastAPI(
    lifespan=lifespan,
    openapi_tags=[
        {
            "name": "Public",
            "description": "不需要 token，任何人皆可呼叫。",
        },
        {
            "name": "User",
            "description": "需要**一般使用者 token**（`POST /SSO/login` 以 `realm_name=dhtsolution`、`client_id=raptor` 取得）。",
        },
        {
            "name": "Admin",
            "description": "需要 **admin token**（`POST /SSO/login` 以 `realm_name=master`、`client_id=admin-cli` 取得）。",
        },
        {
            "name": "System",
            "description": "供服務間（server-to-server）呼叫使用，需要有效的 Bearer token。",
        },
        {
            "name": "Debug",
            "description": "健康檢查，確認服務連線狀態。",
        },
    ],
)

app.include_router(login.router,      prefix="/SSO")
app.include_router(account.router,    prefix="/SSO")
app.include_router(creat_user.router, prefix="/admin/keycloak")
app.include_router(group.router,      prefix="/keycloak")
app.include_router(health.router,     prefix="/keycloak")
app.include_router(path.router,       prefix="/API",   tags=["User"],          dependencies=[Depends(APIsecurity.authenticate)])
app.include_router(auth.router,       prefix="/auth")


@app.get("/")
def root():
    return RedirectResponse(url="/docs")
