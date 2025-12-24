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
                payload=metadata.model_dump()
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
                        payload=metadata.model_dump(),
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
                    payload=metadata.model_dump(),
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

    async def reactivate_metadata(self, asset_path: str, version_id: str, branch: str):
        """
        Reactivate an asset's metadata in the vector store.

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
                        payload={"status": "active"},
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
                    payload={"status": "active"},
                    points=Filter(
                        must=[
                            FieldCondition(key="asset_path", match=MatchValue(value=asset_path)),
                            FieldCondition(key="version_id", match=MatchValue(value=version_id)),
                            # FieldCondition(key="branch", match=MatchValue(value=branch))  # Uncomment if branch condition is needed
                        ]
                    )
                )
        except Exception as e:
            logger.error(f"Failed to reactivate metadata: {str(e)}")

    async def delete_metadata(self, asset_path: str, version_id: str, branch: str):
        """
        Delete an asset's metadata in the vector store.

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

    async def clone_point(
        self, 
        source_path: str, 
        source_version: str, 
        target_path: str, 
        target_version: str, 
        branch: str,
    ):
        """
        Clone points in the vector store from source to target.
        Only changes the asset_path and version_id in the payload,
        while reusing the vector and other fields from the source point.
        Args:
            source_path (str): The asset path of the source point.
            source_version (str): The version ID of the source point.
            target_path (str): The asset path of the target point.
            target_version (str): The version ID of the target point.
            branch (str): The branch for the target point.
        """
        file_type = f"{source_path.split('/', 1)[0]}"
        collection_name = self.file_type_to_collection.get(file_type)
        if not collection_name:
            raise ValueError(f"Invalid file type for path: {source_path}")

        offset = None
        cloned_count = 0

        while True:
            # 1. Fetch all points of the source asset in batches (100 per batch)
            points, next_page_offset = await self.client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="asset_path", match=MatchValue(value=source_path)),
                        FieldCondition(key="version_id", match=MatchValue(value=source_version)),
                        # FieldCondition(key="branch", match=MatchValue(value=branch))  # Uncomment if branch condition is needed
                    ]
                ),
                limit=100, # 100 points per batch
                offset=offset,
                with_vectors=True
            )

            if not points:
                break

            new_points = []
            for p in points:
                # 2. Copy Payload and update key information
                # Note: Retain original paragraph content (text), paragraph number (chunk_idx), etc.
                new_payload = dict(p.payload)
                new_payload.update({
                    "asset_path": target_path,
                    "version_id": target_version,
                    "status": "active",
                    # "branch": branch,
                })

                # 3. Create new point (retain original vector)
                new_points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=p.vector,
                    payload=new_payload
                ))

            # 4. Batch write new points
            if new_points:
                await self.client.upsert(
                    collection_name=collection_name,
                    points=new_points
                )
                cloned_count += len(new_points)

            offset = next_page_offset
            if offset is None:
                break

        logger.info(f"Successfully cloned {cloned_count} points from {source_path} to {target_path}")