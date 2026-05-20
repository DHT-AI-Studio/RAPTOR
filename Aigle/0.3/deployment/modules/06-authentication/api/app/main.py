# keycloak/app/main.py

from fastapi import FastAPI, HTTPException, Form, Response, Request, Query
from fastapi.responses import RedirectResponse
from fastapi import Depends
from contextlib import asynccontextmanager
import asyncio
import logging
import requests
import json
#from scripts.keycloak_function import KeycloakUser
from typing import Optional
import uuid

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        import sync_resources
        await asyncio.to_thread(
            sync_resources.sync,
            apply=True, verbose=False, retries=6, retry_delay=10
        )
    except Exception as e:
        logger.warning(f"Keycloak resource sync failed (non-fatal): {e}")
    yield


app = FastAPI(lifespan=lifespan)

# ---------- Include the contact router ----------

from app.routers import login
app.include_router(login.router, prefix="/SSO")
from app.routers import creat_user
app.include_router(creat_user.router, prefix="/admin", tags=["administrator"])
from app.routers import keycloak_health
app.include_router(keycloak_health.router)
# from app.routers import email_system 
# app.include_router(email_system.router, tags=["mail-system"],prefix="/mail-system")  # prefix 可以加在路由前面

from app.dependencies.security import APIsecurity
from app.routers import path
app.include_router(path.router, prefix="/API", tags=["security test"], dependencies=[Depends(APIsecurity.authenticate)])

from app.routers import auth
app.include_router(auth.router)


# @app.middleware("http")
# async def add_token_to_header(request: Request, call_next):
#     token = request.cookies.get("access_token")

#     if token:
#         # 強制加入 Authorization header
#         request.headers.__dict__["_list"].append(
#             (
#                 b"authorization",
#                 f"Bearer {token}".encode()
#             )
#         )

#     response = await call_next(request)
#     return response

'''
@app.post("/register", tags=["login"])
def register(
    username: str = Form(...),
    password: str = Form(...),
    email: str = Form(...),
    firstName: str = Form(...),
    lastName: str = Form(...),
    realm_name: str = Query("raptor", enum=["raptor"])  #realm_name: str = Form(None) # optional
):
    try:
        # 建立一個 dict，動態決定要傳哪些參數
        user_kwargs = {
            "username": username,
            "user_password": password,
            "email": email,
            "firstName": firstName,
            "lastName": lastName
        }

        if realm_name and realm_name.strip():  # 只有 realm_name 非空字串才傳
            user_kwargs["realm_name"] = realm_name

        response = KeycloakUser.creat_user(**user_kwargs)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if response.status_code == 201:
        return "✅ Registration successful"
        raise HTTPException(status_code=409, detail="⚠️ User already exists")
    else:
        raise HTTPException(status_code=500, detail=f"{response.text}")

# 簡單 session 存儲
sessions: dict[str, str] = {}  # session_id -> access_token

# =============================
# Login
@app.post("/login", tags=["login"])
def login(response: Response,
          username: str = Form(...),
          password: str = Form(...),
          realm_name: str = Query("raptor", enum=["raptor"]),   #realm_name: str = Form(None),
          client_id: str = Query("base-client", enum=["base-client"])   #client_id: str = Form(None)
          ):
    try:
        kwargs = {"username": username, "password": password}
        if realm_name and realm_name.strip():
            kwargs["realm_name"] = realm_name.strip()
        if client_id and client_id.strip():
            kwargs["client_id"] = client_id.strip()

        resp = KeycloakUser.user_login(**kwargs)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    #if resp.status_code == 202:
    #    return "✅ Login successful: 💬 please change your password and log in again."
    if resp.status_code == 400:
        if resp.json().get("error_description") == "Account is not fully set up":
            raise HTTPException(status_code=202, detail=f"Unable to obtain token: {resp.text}")
        raise HTTPException(status_code=resp.status_code, detail=f"Unable to obtain token: {resp.text}")
    
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"Unable to obtain token: {resp.text}")

    access_token = resp.json().get("access_token")
    # 產生 session_id
    session_id = str(uuid.uuid4())
    sessions[session_id] = access_token

    # 將 session_id 放到 cookie
    response.set_cookie(key="session_id", value=session_id, httponly=True)

    # 將 JWT access_token 也放到 cookie
    response.set_cookie(key="access_token", value=access_token, httponly=True, samesite="lax", secure=False)

    ## 將 realm_name, client_name, username 放到 cookie
    response.set_cookie(key="realm_name", value=realm_name, httponly=True, samesite="lax", secure=False)
    response.set_cookie(key="client_name", value=client_id, httponly=True, samesite="lax", secure=False)
    response.set_cookie(key="username", value=username, httponly=True, samesite="lax", secure=False)

    return "✅🧑‍💻 User logging successful"

# =============================
# Logout
@app.post("/logout", tags=["user home"])
def logout(request: Request, response: Response):
    # 從 cookie 取得 session_id
    session_id = request.cookies.get("session_id")
    if not session_id or session_id not in sessions:
        raise HTTPException(status_code=400, detail="No active session")

    access_token = sessions[session_id]
    try:
        resp = KeycloakUser.user_logout(access_token)
        if resp.status_code == 204:
            # 登出成功後刪除 session 並清 cookie
            del sessions[session_id]
            response.delete_cookie("session_id")
            return {"message": "Logout successful"}
        else:
            raise HTTPException(status_code=resp.status_code, detail=f"Logout failed: {resp.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Delete accoount
@app.delete("/delete_accoount", tags=["user home"])
def logout(request: Request, response: Response):
    # 從 cookie 取得 session_id
    session_id = request.cookies.get("session_id")
    if not session_id or session_id not in sessions:
        raise HTTPException(status_code=400, detail="No active session")

    access_token = sessions[session_id]
    try:
        resp = KeycloakUser.delete_own_user(access_token)
        if resp.status_code == 204:
            # 登出成功後刪除 session 並清 cookie
            del sessions[session_id]
            response.delete_cookie("session_id")
            return {"message": "Delete accoount successful"}
        else:
            raise HTTPException(status_code=resp.status_code, detail=f"Delete accoount failed: {resp.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    



@app.post("/change-password", tags=["user home"])
def api_reset_temporary_password(
    request: Request,
    username: str = Form(...),
    new_password: str = Form(...),
    realm: str = Query("raptor", enum=["raptor"]),  # realm: str = Form(...),
):
    """
    使用者登入後 → 送出 request → FastAPI 讀 cookie 裡的 token
    呼叫 Keycloak Admin API 將 temporary password 改為正式密碼
    """

    # 從 cookie 讀取 access_token
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="No token found in cookie")

    try:
        ok = KeycloakUser.change_temporary_password_to_permanent(
            realm_name=realm,
            user_token=token,  # 用 cookie 裡的 token
            username=username,
            new_password=new_password
        )

        if ok:
            return {"status": "success", "message": "Password updated successfully"}

        raise HTTPException(status_code=400, detail="Failed to update password")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
'''
@app.get("/")
def root():
    return RedirectResponse(url="/docs")
