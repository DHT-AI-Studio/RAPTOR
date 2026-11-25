from collections.abc import Callable

from app.core.exceptions import ValidationError
from app.schemas import DEFAULT_EXTRACTORS, PayloadSchema


class PayloadSchemaManager:
    def __init__(self):
        self.schemas: dict[str, PayloadSchema] = DEFAULT_EXTRACTORS.copy()

    def register_schema(self, schema: PayloadSchema):
        self.schemas[schema.name] = schema

    def get_schema(self, name: str) -> PayloadSchema:
        if name not in self.schemas:
            raise ValidationError(f"Unknown payload schema: '{name}'")
        return self.schemas[name]

    def get_extractor(self, name: str) -> Callable:
        return self.get_schema(name).content_extractor

    def get_bm25_fields(self, name: str) -> tuple[list[str], list[str]]:
        schema = self.get_schema(name)
        return schema.bm25_fields, schema.nested_paths
