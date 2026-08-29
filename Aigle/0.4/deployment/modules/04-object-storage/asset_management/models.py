from datetime import datetime
from typing import List, Tuple, Any, Dict, Type, Optional, Union
from enum import Enum
from pydantic import BaseModel

class MediaType(Enum):
    VIDEO = "video"
    IMAGE = "image"
    AUDIO = "audio"
    DOCUMENT = "document"
    OTHER = "other"

class ExistenceInfo(BaseModel):
    exists: bool = False
    message: str = ""
    existing_asset_path: Optional[str] = None
    existing_version_id: Optional[str] = None

class AssetMetadata(BaseModel):
    asset_path: str
    version_id: str
    primary_filename: str
    associated_filenames: List[Tuple[str, str]] 
    upload_date: datetime
    archive_date: Optional[Union[datetime, str]] = None
    destroy_date: Optional[Union[datetime, str]] = None
    user: str
    branch: str
    status: str
    checksum: str
    existence_info: ExistenceInfo = ExistenceInfo()

class AssetMetadataResponse(BaseModel):
    asset_path: str
    version_id: str
    primary_filename: str
    associated_filenames: List[Tuple[str, str]] 
    upload_date: datetime
    archive_date: Optional[Union[datetime, str]] = None
    destroy_date: Optional[Union[datetime, str]] = None
    status: str

class Token(BaseModel):
    access_token: str
    token_type: str
    username: str
    branch: str

class User(BaseModel):
    user_id: str
    branch_id: str

def model_to_response(
    model_obj: BaseModel, 
    response_model: Type[BaseModel],
    field_map: Dict[str, Any] = None
) -> BaseModel:
    """
    Convert a Pydantic model object to another Pydantic response model dynamically.
    
    Args:
        model_obj: The Pydantic model instance to convert.
        response_model: The Pydantic model class for response.
        field_map: Optional mapping for custom transformations: 
                   {response_field_name: callable or source_field_name}
    
    Returns:
        An instance of response_model with fields populated.
    """
    data = model_obj.dict()
    field_map = field_map or {}
    response_data = {}

    for field in response_model.__fields__:
        if field in field_map:
            mapping = field_map[field]
            if callable(mapping):
                response_data[field] = mapping(model_obj)
            elif isinstance(mapping, str):
                response_data[field] = data.get(mapping)
        else:
            response_data[field] = data.get(field)

    return response_model(**response_data)