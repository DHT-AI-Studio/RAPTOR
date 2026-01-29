from fastapi import APIRouter, HTTPException, Body, Depends, Form, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Annotated
from keycloak.scripts.keycloak_user import KeycloakUser
from keycloak.scripts.keycloak_master import KeycloakMaster

router = APIRouter()

bearer_scheme = HTTPBearer(auto_error=True)  # 這裡 auto_error=True，沒有 token 就 401

class UserCreate(BaseModel):
    username: str = "john"
    password: str = "TempPass123!"
    email: str = "john@example.com"
    firstName: str = "John"
    lastName: str = "Doe"

# @router.post("/keycloak/create-user")
# def create_user_api(
#     user_data: Annotated[
#         UserCreate,
#         Body(
#             examples=[
#                 {
#                     "username": "john",
#                     "password": "TempPass123!",
#                     "email": "john@example.com",
#                     "firstName": "John",
#                     "lastName": "Doe"
#                 }  
#             ]
#         ),
#     ],
#     realm_name: str= Query("dhtsolution", enum=["dhtsolution"]),
# ):
#     try:
#         admin_token = KeycloakMaster.MasterToken()
#         if not admin_token:
#             raise HTTPException(status_code=500, detail="Cannot get Keycloak admin token")

#         user_dict = user_data.dict()
#         #realm_name = user_dict.pop("realm_name")  # 取出 realm_name
#         response = KeycloakUser.create_user(realm_name, admin_token, user_dict)

#         if response.status_code == 201:
#             return {
#                 "message": "User created successfully. Verification email sent.",
#                 "username": user_data.username,
#                 "email": user_data.email
#             }
#         else:
#             try:
#                 detail = response.json()
#             except:
#                 detail = response.text
#             raise HTTPException(status_code=response.status_code, detail=detail)

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
    

@router.post("/keycloak/create-user")
def create_user_api(
    user_data: Annotated[
        UserCreate,
        Body(
            examples=[
                {
                    "username": "john",
                    "password": "TempPass123!",
                    "email": "john@example.com",
                    "firstName": "John",
                    "lastName": "Doe"
                }  
            ]
        ),
    ],
    realm_name: str = Query("dhtsolution", enum=["dhtsolution"]),
    token: HTTPAuthorizationCredentials = Depends(bearer_scheme)  # <- 這裡注入 token
):
    try:
        admin_token = token.credentials  # <- 正確取得 token 字串
        if not admin_token:
            raise HTTPException(status_code=500, detail="Cannot get Keycloak admin token")

        user_dict = user_data.dict()
        response = KeycloakUser.create_user(realm_name, admin_token, user_dict)

        if response.status_code == 201:
            return {
                "message": "User created successfully. Verification email sent.",
                "username": user_data.username,
                "email": user_data.email
            }
        else:
            try:
                detail = response.json()
            except:
                detail = response.text
            raise HTTPException(status_code=response.status_code, detail=detail)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))