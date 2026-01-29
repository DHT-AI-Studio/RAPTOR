"""
API router for authentication, providing an endpoint to obtain a JWT.
"""
from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

from app.api.dependencies import get_auth_service
from app.services.auth_service import AuthService

router = APIRouter()

ACCESS_TOKEN_EXPIRE_MINUTES = 30


class RegistrationRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=8, max_length=128)


@router.post(
    "/register",
    tags=["authentication"],
    summary="Register a new gateway user",
    status_code=status.HTTP_201_CREATED,
)
async def register_user(
    payload: RegistrationRequest,
    auth_service: AuthService = Depends(get_auth_service),
):
    """Register a new gateway user with a hashed password.

    **Usage:**
    ```python
    url = f"{base_url}/api/v1/auth/register"
    payload = {
        "username": username,
        "email": email,
        "password": password
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    result = response.json()
    ```
    """
    user_info = await auth_service.register_user(payload.username, payload.password)
    return {"username": user_info["username"]}


@router.post("/login", tags=["authentication"], summary="Obtain a gateway JWT")
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    Provides a JWT access token for valid user credentials.

    **Usage:**
    ```python
    url = f"{base_url}/api/v1/auth/login"
    data = {
        "username": username,
        "password": password
    }
    response = requests.post(url, data=data)
    response.raise_for_status()
    token = response.json().get("access_token")
    ```
    """
    user = await auth_service.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth_service.create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )

    return {"access_token": access_token, "token_type": "bearer"}
