import os
from typing import List, Dict, Any
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
import aiohttp
import logging

from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.routing import Mount
from starlette.types import Receive, Scope, Send

import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.fastmcp import Context
from mcp.server.session import ServerSession
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

# ----------------------------------
# Base URL for Asset Management API
# ----------------------------------
base_url = os.getenv("BASE_URL", "http://YOUR_IP_ADDRESS:8000")

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


# ------------------------------
# Token acquisition
# ------------------------------
def get_bearer_token(ctx):
    # Check if 'Authorization' header is present
    auth_header = ctx.request_context.request.headers.get("authorization")

    if auth_header:
        if auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1]
            return token
        else:
            raise ValueError("Invalid Authorization header format")
    else:
        raise ValueError("Authorization header missing")


# ------------------------------
# MCP Server
# ------------------------------
server = Server("asset-management-server")


# ------------------------------
# Tool implementations
# ------------------------------
async def upload_file_tool(
    token: str,
    primary_file: str,
    associated_files: List[str] = [],
    archive_ttl: int = 30,
    destroy_ttl: int = 30,
) -> Dict:
    """
    Upload a primary file and optional associated files

    Args:
        token (str): token to authenticate
        primary_file (str): path to the primary file
        associated_files (List[str], optional): paths to associated files. Defaults to [].
        archive_ttl (int, optional): archive ttl in days. Defaults to 30.
        destroy_ttl (int, optional): destroy ttl in days. Defaults to 30.

    Returns:
        Dict: response from the server
    """
    logger.info(f"Starting upload of {primary_file} with {len(associated_files)} associated files ...")

    async with aiohttp.ClientSession() as session:
        form = aiohttp.FormData()
        form.add_field("primary_file", open(primary_file, "rb"), filename=os.path.basename(primary_file))
        for af in associated_files:
            form.add_field("associated_files", open(af, "rb"), filename=os.path.basename(af))
        form.add_field("archive_ttl", str(archive_ttl))
        form.add_field("destroy_ttl", str(destroy_ttl))

        async with session.post(
            f"{base_url}/fileupload",
            headers={"Authorization": f"Bearer {token}"},
            data=form,
        ) as resp:
            result = await resp.json()
        
    logger.info(f"Upload of {primary_file} completed")

    return result


async def add_associated_files_tool(
    token: str,
    asset_path: str,
    associated_files: List[str],
    primary_version_id: str,
) -> Dict:
    
    """
    Add associated files to an existing primary asset at a specific version

    Args:
        token (str): token to authenticate
        asset_path (str): Path of the existing asset (e.g., 'reports/annual_report')
        associated_files (List[str]): List of paths to associated files
        primary_version_id (str): Specific version_id to update.

    Returns:
        Dict: response from the server
    """

    logger.info(f"Adding associated files to {asset_path}/{primary_version_id} ...")

    async with aiohttp.ClientSession() as session:
        form = aiohttp.FormData()
        for associated_file in associated_files:
            form.add_field("associated_files", open(associated_file, "rb"), filename=associated_file)
        form.add_field("primary_version_id", primary_version_id)

        async with session.post(
            f"{base_url}/add-associated-files/{asset_path}",
            headers={"Authorization": f"Bearer {token}"},
            data=form
        ) as resp:
            result = await resp.json()

    logger.info(f"Adding associated files to {asset_path}/{primary_version_id} completed")

    return result


async def file_download_tool(
    token: str,
    asset_path: str, 
    version_id: str
) -> Dict:
    """
    Download a file and its associated files, each file is returned as a base64 string accompanied by the presigned url

    Args:
        token (str): token to authenticate
        asset_path (str): path to the asset
        version_id (str): version id of the asset

    Returns:
        Dict: response from the server
    """
    logger.info(f"Downloading {asset_path}/{version_id} ...")

    # Since the base64 encoded file string may be too long, we don't return it.
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{base_url}/filedownload/{asset_path}/{version_id}?return_file_content=False",
            headers={"Authorization": f"Bearer {token}"},
        ) as resp:
            result = await resp.json()

    logger.info(f"Download of {asset_path}/{version_id} completed")

    return result


async def list_file_versions_tool(
    token: str,
    asset_path: str, 
    primary_filename: str
) -> Dict:
    """
    List file versions

    Args:
        asset_path (str): path to the asset
        primary_filename (str): primary filename of the asset

    Returns:
        List[Dict]: response from the server
    """
    logger.info(f"Listing versions of {asset_path}/{primary_filename} ...")

    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{base_url}/fileversions/{asset_path}/{primary_filename}",
            headers={"Authorization": f"Bearer {token}"},
        ) as resp:
            result = await resp.json()

    logger.info(f"Listing versions of {asset_path}/{primary_filename} completed")

    return {"result": result}


async def file_archive_tool(
    token: str,
    asset_path: str, 
    version_id: str,
) -> Dict:
    
    """
    Archive a file

    Args:
        token (str): token to authenticate
        asset_path (str): path to the asset
        version_id (str): version id of the asset

    Returns:
        Dict: response from the server
    """
    logger.info(f"Archiving {asset_path}/{version_id} ...")

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/filearchive/{asset_path}/{version_id}",
            headers={"Authorization": f"Bearer {token}"}
        ) as resp:
            result = await resp.json()

    logger.info(f"Archiving {asset_path}/{version_id} completed")

    return result


async def file_delete_tool(
    token: str,
    asset_path: str, 
    version_id: str,
) -> Dict:

    """
    Delete a file

    Args:
        token (str): token to authenticate
        asset_path (str): path to the asset
        version_id (str): version id of the asset

    Returns:
        Dict: response from the server
    """
    logger.info(f"Deleting {asset_path}/{version_id} ...")

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/delfile/{asset_path}/{version_id}",
            headers={"Authorization": f"Bearer {token}"}
        ) as resp:
            result = await resp.json()

    logger.info(f"Deleting {asset_path}/{version_id} completed")

    return result


# ------------------------------
# MCP call_tool dispatcher
# ------------------------------
@server.call_tool()
async def handle_call_tool(name: str, arguments: dict[str, Any]) -> Any:
    token = get_bearer_token(server) # get jwt token

    """Dispatch tool calls to the corresponding function."""
    try:
        if name == "upload_file":
            return await upload_file_tool(token=token, **arguments)
        elif name == "add_associated_files":
            return await add_associated_files_tool(token=token, **arguments)
        elif name == "file_download":
            return await file_download_tool(token=token, **arguments)
        elif name == "list_file_versions":
            return await list_file_versions_tool(token=token, **arguments)
        elif name == "file_archive":
            return await file_archive_tool(token=token, **arguments)
        elif name == "file_delete":
            return await file_delete_tool(token=token, **arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")
    except Exception as e:
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]


# ------------------------------
# Define available tools
# ------------------------------
@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    return [
        types.Tool(
            name="upload_file",
            description="""
            Upload a primary file and optional associated files

            Args:
                primary_file (str): path to the primary file
                associated_files (List[str], optional): paths to associated files. Defaults to [].
                archive_ttl (int, optional): archive ttl in days. Defaults to 30.
                destroy_ttl (int, optional): destroy ttl in days. Defaults to 30.

            Returns:
                Dict: Information about the uploaded asset, with keys:
                    - asset_path (str): Path of the uploaded asset.
                    - version_id (str): Version ID assigned to the upload.
                    - primary_filename (str): Name of the primary file.
                    - associated_filenames (List[List[str]]): List of associated files and their version IDs, e.g.,
                    [["test.json", "version id"], ["test.jpg", "version id"]].
                    - upload_date (str): Upload timestamp in ISO 8601 format.
                    - archive_date (str): Scheduled archive timestamp in ISO 8601 format.
                    - destroy_date (str): Scheduled destroy timestamp in ISO 8601 format.
                    - status (str): Current status of the asset (e.g., "active").
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "primary_file": {"type": "string", "description": "path to the primary file"},
                    "associated_files": {"type": "array", "items": {"type": "string"}, "description": "paths to associated files. Defaults to []."},
                    "archive_ttl": {"type": "integer", "description": "archive ttl in days. Defaults to 30."},
                    "destroy_ttl": {"type": "integer", "description": "destroy ttl in days. Defaults to 30."},
                },
                "required": ["primary_file"],
            },
        ),
        types.Tool(
            name="add_associated_files",
            description="""
            Add associated files to an existing primary asset at a specific version

            Args:
                asset_path (str): Path of the existing asset (e.g., 'reports/annual_report')
                associated_files (List[str]): List of paths to associated files
                primary_version_id (str): Specific version_id to update.

            Returns:
                Dict: Information about the updated asset, with keys:
                    - asset_path (str): Path of the updated asset.
                    - version_id (str): Version ID assigned to the original upload, same as primary_version_id.
                    - primary_filename (str): Name of the primary file.
                    - associated_filenames (List[List[str]]): List of associated files and their version IDs, e.g.,
                    [["test.json", "version id"], ["test.jpg", "version id"]].
                    - upload_date (str): Upload timestamp in ISO 8601 format.
                    - archive_date (str): Scheduled archive timestamp in ISO 8601 format.
                    - destroy_date (str): Scheduled destroy timestamp in ISO 8601 format.
                    - status (str): Current status of the asset (e.g., "active").
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "asset_path": {"type": "string", "description": "path to the asset"},
                    "associated_files": {"type": "array", "items": {"type": "string"}, "description": "paths to associated files. Defaults to []."},
                    "primary_version_id": {"type": "string", "description": "version id of the primary file"},
                },
                "required": ["asset_path", "associated_files"],
            },
        ),
        types.Tool(
            name="file_download",
            description="""
            Download a file and its associated files, each file is returned as a presigned url.

            Args:
                asset_path (str): path to the asset
                version_id (str): version id of the asset

            Returns:
                Dict: Response object containing metadata and signed download URLs.
                    - metadata (Dict): Information about the downloaded asset.
                        - asset_path (str): Path of the downloaded asset.
                        - version_id (str): Version ID assigned to the original upload.
                        - primary_filename (str): Name of the primary file.
                        - associated_filenames (List[List[str]]): List of associated files and their hashes,
                        e.g., [["test.json", "hash1"], ["test.jpg", "hash2"]].
                        - upload_date (str): Upload timestamp in ISO 8601 format.
                        - archive_date (str): Scheduled archive timestamp in ISO 8601 format.
                        - destroy_date (str): Scheduled destroy timestamp in ISO 8601 format.
                        - status (str): Current status of the asset (e.g., "active").
                    - primary_file (Dict): Download information of the primary file.
                        - filename (str): File name.
                        - content_type (str): Content type of the file.
                        - version_id (str): Version ID assigned to the original upload.
                        - url (str): Pre-signed download URL.
                    - associated_file_1..N (Dict): Download information of each associated file.
                        - filename (str): File name.
                        - content_type (str): Content type of the file.
                        - version_id (str): Version ID assigned to the original upload.
                        - url (str): Pre-signed download URL.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "asset_path": {"type": "string", "description": "path to the asset"},
                    "version_id": {"type": "string", "description": "version id of the asset"},
                },
                "required": ["asset_path", "version_id"],
            },
        ),
        types.Tool(
            name="list_file_versions",
            description=    """
            List file versions

            Args:
                asset_path (str): path to the asset
                primary_filename (str): primary filename of the asset

            Returns:
                Dict: Response containing a list of file versions with signed download URLs.
                    - result (List[Dict]): List of available versions for the requested file.
                        Each item contains:
                        - key (str): Object key (full path of the file in storage).
                        - version_id (str): Unique identifier for the file version.
                        - last_modified (str): ISO 8601 timestamp when this version was last modified.
                        - url (str): Pre-signed URL for downloading this version of the file.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "asset_path": {"type": "string", "description": "path to the asset"},
                    "primary_filename": {"type": "string", "description": "primary filename of the asset"},
                },
                "required": ["asset_path", "primary_filename"],
            },
        ),
        types.Tool(
            name="file_archive",
            description="""
            Archive a file

            Args:
                asset_path (str): path to the asset
                version_id (str): version id of the asset

            Returns:
                Dict: Information about the updated asset, with keys:
                    - asset_path (str): Path of the updated asset.
                    - version_id (str): Version ID assigned to the original upload, same as version_id.
                    - primary_filename (str): Name of the primary file.
                    - associated_filenames (List[List[str]]): List of associated files and their version IDs, e.g.,
                    [["test.json", "version id"], ["test.jpg", "version id"]].
                    - upload_date (str): Upload timestamp in ISO 8601 format.
                    - archive_date (str): Scheduled archive timestamp in ISO 8601 format.
                    - destroy_date (str): Scheduled destroy timestamp in ISO 8601 format.
                    - status (str): Current status of the asset (e.g., "archived").
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "asset_path": {"type": "string", "description": "path to the asset"},
                    "version_id": {"type": "string", "description": "version id of the asset"},
                },
                "required": ["asset_path", "version_id"],
            },
        ),
        types.Tool(
            name="file_delete",
            description=    """
            Delete a file

            Args:
                asset_path (str): path to the asset
                version_id (str): version id of the asset

            Returns:
                Dict: Information about the updated asset, with keys:
                    - asset_path (str): Path of the updated asset.
                    - version_id (str): Version ID assigned to the original upload, same as version_id.
                    - primary_filename (str): Name of the primary file.
                    - associated_filenames (List[List[str]]): List of associated files and their version IDs, e.g.,
                    [["test.json", "version id"], ["test.jpg", "version id"]].
                    - upload_date (str): Upload timestamp in ISO 8601 format.
                    - archive_date (str): Scheduled archive timestamp in ISO 8601 format.
                    - destroy_date (str): Scheduled destroy timestamp in ISO 8601 format.
                    - status (str): Current status of the asset (e.g., "archived").
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "asset_path": {"type": "string", "description": "path to the asset"},
                    "version_id": {"type": "string", "description": "version id of the asset"},
                },
                "required": ["asset_path", "version_id"],
            },
        ),
    ]


# ------------------------------
# Streamable HTTP session manager
# ------------------------------
# event_store = None  # For demonstration, can implement InMemoryEventStore if needed

session_manager = StreamableHTTPSessionManager(
    app=server,
    # event_store=event_store,
    # json_response=False,
)


async def handle_streamable_http(scope: Scope, receive: Receive, send: Send) -> None:
    await session_manager.handle_request(scope, receive, send)


@asynccontextmanager
async def lifespan(app: Starlette) -> AsyncIterator[None]:
    async with session_manager.run():
        logger.info("Streamable HTTP server started!")
        try:
            yield
        finally:
            logger.info("Server shutting down...")


# ------------------------------
# ASGI application
# ------------------------------
starlette_app = Starlette(
    debug=True,
    routes=[Mount("/mcp", app=handle_streamable_http)],
    lifespan=lifespan,
)

starlette_app = CORSMiddleware(
    starlette_app,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE"],
    expose_headers=["Mcp-Session-Id"],
)


# ------------------------------
# Run server
# ------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(starlette_app, host="0.0.0.0", port=8000)
