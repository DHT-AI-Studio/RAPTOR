import uuid
from datetime import timedelta
import pytest
from jose import jwt
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env.test"))

from asset_management.endpoints import create_access_token, SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES


def test_create_access_token_basic():
    # test parameters
    data = {"username": "testuser", "branch": "testbranch", "permissions": ["admin"]}
    expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    token = create_access_token(data, expires_delta=expires)
    assert token is not None
    assert isinstance(token, str)
    
    # decode token
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    
    # verify payload contents
    assert payload["username"] == "testuser"
    assert payload["branch"] == "testbranch"
    assert payload["permissions"] == ["admin"]
    
    assert "exp" in payload
    assert "iat" in payload
    assert "jti" in payload

    # exp > iat
    assert payload["exp"] > payload["iat"]
    
    # jti is a valid UUID
    try:
        uuid_obj = uuid.UUID(payload["jti"])
        assert str(uuid_obj) == payload["jti"]
    except ValueError:
        pytest.fail("jti is not a valid UUID")

    # Ensure the time difference between "exp" and "iat" is exactly ACCESS_TOKEN_EXPIRE_MINUTES
    time_diff = payload["exp"] - payload["iat"]
    assert time_diff == ACCESS_TOKEN_EXPIRE_MINUTES * 60, f"Expected {ACCESS_TOKEN_EXPIRE_MINUTES * 60} seconds, but got {time_diff} seconds"