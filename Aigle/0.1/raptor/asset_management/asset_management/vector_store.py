from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue
import uuid
import logging

from .models import AssetMetadata
from .config import settings


logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        self.client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        self.collection_name = ["documents", "audios", "videos", "images"]
        self.file_type_to_collection = {
            "document": "documents",
            "audio": "audios",
            "video": "videos",
            "image": "images"
        } 
        self._initialized = False

    async def initialize(self):
        """
        Initialize the vector store by checking if all collections exist.
        If not, create them with the default vector config.
        """
        
        if self._initialized:
            return
        
        self._initialized = True
        for collection in self.collection_name:
            try:
                await self.client.get_collection(collection)
            except Exception:
                await self.client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
                )

    async def close(self):
        """
        Close the database connection pool.
        """
        if self.client:
            await self.client.close()
            self._initialized = False
            logger.info("VectorStore closed")

    async def save_or_update_metadata(self, metadata: AssetMetadata):
        """
        Save or update an AssetMetadata object to the vector store.
        The object is stored in a collection based on the file type of the asset.

        This method is a placeholder, just testing the connection between the app and the vector store.
        In a real implementation, you would extract meaningful vectors from the asset and other relevant fields, 
        then combine them with the AssetMetadata object to create a PointStruct object.

        Args:
            metadata: The AssetMetadata object to be saved.

        Raises:
            ValueError: If the file type is not valid.
        """

        file_type = f"{metadata.asset_path.split('/', 1)[0]}"
        if not file_type in self.file_type_to_collection.keys():
            raise ValueError(f"Invalid file type: {file_type}")
        
        collection_name = self.file_type_to_collection[file_type]

        check_filter = Filter(
            must=[
                FieldCondition(key="asset_path", match=MatchValue(value=metadata.asset_path)),
                FieldCondition(key="version_id", match=MatchValue(value=metadata.version_id)),
                FieldCondition(key="branch", match=MatchValue(value=metadata.branch)),
            ]
        )

        try:
            # First, check if a record with the same asset path, version id, and branch already exists.
            existing = await self.client.scroll(
                collection_name=collection_name if isinstance(collection_name, str) else collection_name[0],
                scroll_filter=check_filter,
                limit=1
            )

            if existing[0]:  # If it does, update the existing record
                await self.update_metadata(metadata)
                logger.info(f"Metadata for {metadata.asset_path} updated successfully.")
                return

            # If not, create a new record
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=[0.0] * 1024,  # Placeholder vector, replace with actual vector extraction logic
                payload=metadata.to_dict()
            )

            if isinstance(collection_name, list):
                for collection in collection_name:
                    await self.client.upsert(
                        collection_name=collection,
                        points=[point]
                    )
            else:
                await self.client.upsert(
                    collection_name=collection_name,
                    points=[point]
                )
            logger.info(f"Metadata for {metadata.asset_path} saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save metadata: {str(e)}")

    async def update_metadata(self, metadata: AssetMetadata):
        """
        Update the AssetMetadata object with the given asset_path and version_id.
        The object is updated in a collection based on the file type of the asset.

        This method is a placeholder, just testing the connection between the app and the vector store.
        In a real implementation, you would extract meaningful vectors from the asset and other relevant fields, 
        then combine them with the AssetMetadata object to create a PointStruct object.

        Args:
            metadata: The AssetMetadata object to be updated.

        Raises:
            ValueError: If the file type is not valid.
        """
        asset_path = metadata.asset_path
        version_id = metadata.version_id
        branch = metadata.branch

        file_type = f"{asset_path.split('/', 1)[0]}"
        if file_type not in self.file_type_to_collection.keys():
            raise ValueError(f"Invalid file type: {file_type}")
        
        collection_name = self.file_type_to_collection[file_type]
        try:
            if isinstance(collection_name, list):
                for collection in collection_name:
                    await self.client.set_payload(
                        collection_name=collection,
                        payload=metadata.to_dict(),
                        points=Filter(
                            must=[
                                FieldCondition(key="asset_path", match=MatchValue(value=asset_path)),
                                FieldCondition(key="version_id", match=MatchValue(value=version_id)),
                                FieldCondition(key="branch", match=MatchValue(value=branch))
                            ]
                        )
                    )
            elif isinstance(collection_name, str):
                await self.client.set_payload(
                    collection_name=collection_name,
                    payload=metadata.to_dict(),
                    points=Filter(
                        must=[
                            FieldCondition(key="asset_path", match=MatchValue(value=asset_path)),
                            FieldCondition(key="version_id", match=MatchValue(value=version_id)),
                            FieldCondition(key="branch", match=MatchValue(value=branch))
                        ]
                    )
                )
        except Exception as e:
            logger.error(f"Failed to update metadata: {str(e)}")

    async def archive_metadata(self, asset_path: str, version_id: str, branch: str):
        """
        Archive an asset's metadata in the vector store.

        Args:
            asset_path (str): path to the asset
            version_id (str): version id of the asset
            branch (str): branch of the asset
        Raises:
            ValueError: If the file type is not valid.

        """
        file_type = f"{asset_path.split('/', 1)[0]}"
        if file_type not in self.file_type_to_collection.keys():
            raise ValueError(f"Invalid file type: {file_type}")
        
        collection_name = self.file_type_to_collection[file_type]
        try:
            if isinstance(collection_name, list):
                for collection in collection_name:
                    await self.client.set_payload(
                        collection_name=collection,
                        payload={"status": "archived"},
                        points=Filter(
                            must=[
                                FieldCondition(key="asset_path", match=MatchValue(value=asset_path)),
                                FieldCondition(key="version_id", match=MatchValue(value=version_id)),
                                # FieldCondition(key="branch", match=MatchValue(value=branch))  # Uncomment if branch condition is needed
                            ]
                        )
                    )
            elif isinstance(collection_name, str):
                await self.client.set_payload(
                    collection_name=collection_name,
                    payload={"status": "archived"},
                    points=Filter(
                        must=[
                            FieldCondition(key="asset_path", match=MatchValue(value=asset_path)),
                            FieldCondition(key="version_id", match=MatchValue(value=version_id)),
                            # FieldCondition(key="branch", match=MatchValue(value=branch))  # Uncomment if branch condition is needed
                        ]
                    )
                )
        except Exception as e:
            logger.error(f"Failed to archive metadata: {str(e)}")

    async def delete_metadata(self, asset_path: str, version_id: str, branch: str):
        """
        Delete an asset's metadata in the vector store.

        Args:
            asset_path (str): path to the asset
            version_id (str): version id of the asset

        Raises:
            ValueError: If the file type is not valid.
        """
        file_type = f"{asset_path.split('/', 1)[0]}"
        if file_type not in self.file_type_to_collection.keys():
            raise ValueError(f"Invalid file type: {file_type}")
        
        collection_name = self.file_type_to_collection[file_type]
        try:
            if isinstance(collection_name, list):
                for collection in collection_name:
                    await self.client.delete(
                        collection_name=collection,
                        points_selector=Filter(
                            must=[
                                FieldCondition(key="asset_path", match=MatchValue(value=asset_path)),
                                FieldCondition(key="version_id", match=MatchValue(value=version_id)),
                                # FieldCondition(key="branch", match=MatchValue(value=branch))  # Uncomment if branch condition is needed
                            ]
                        )
                    )
            elif isinstance(collection_name, str):
                await self.client.delete(
                    collection_name=collection_name,
                    points_selector=Filter(
                        must=[
                            FieldCondition(key="asset_path", match=MatchValue(value=asset_path)),
                            FieldCondition(key="version_id", match=MatchValue(value=version_id)),
                            # FieldCondition(key="branch", match=MatchValue(value=branch))  # Uncomment if branch condition is needed
                        ]
                    )
                )
        except Exception as e:
            logger.error(f"Failed to delete metadata: {str(e)}")
