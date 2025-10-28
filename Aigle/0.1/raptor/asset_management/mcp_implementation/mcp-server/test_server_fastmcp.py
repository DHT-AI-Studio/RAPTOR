from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession
import aiohttp
from typing import List, Dict
import logging
import os


# ----------------------------------
# Base URL for Asset Management API
# ----------------------------------
base_url = os.getenv("BASE_URL", "http://YOUR_IP_ADRESS:8000")

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
mcp = FastMCP(
    name="Asset Management MCP Server", 
    host="0.0.0.0", 
    port=8000
)


# ------------------------------
# Tool implementations
# ------------------------------
@mcp.tool()
async def upload_file(
    ctx: Context[ServerSession, None],
    primary_file: str, 
    associated_files: List[str] = [],
    archive_ttl: int = 30, 
    destroy_ttl: int = 30
) -> Dict:

    """
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
    """
    user_token = get_bearer_token(ctx)

    logger.info(f"Starting upload of {primary_file} with {len(associated_files)} associated files ...")

    async with aiohttp.ClientSession() as session:
        form = aiohttp.FormData()
        form.add_field("primary_file", open(primary_file, "rb"), filename=primary_file)
        for associated_file in associated_files:
            form.add_field("associated_files", open(associated_file, "rb"), filename=associated_file)
        form.add_field("archive_ttl", str(archive_ttl))
        form.add_field("destroy_ttl", str(destroy_ttl))

        async with session.post(
            f"{base_url}/fileupload",
            headers={"Authorization": f"Bearer {user_token}"},
            data=form
        ) as resp:
            result = await resp.json()

    logger.info(f"Upload of {primary_file} completed")

    return result


@mcp.tool()
async def add_associated_files(
    ctx: Context[ServerSession, None],
    asset_path: str,
    associated_files: List[str],
    primary_version_id: str,
) -> Dict:
    
    """
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
    """
    user_token = get_bearer_token(ctx)

    logger.info(f"Adding associated files to {asset_path}/{primary_version_id} ...")

    async with aiohttp.ClientSession() as session:
        form = aiohttp.FormData()
        for associated_file in associated_files:
            form.add_field("associated_files", open(associated_file, "rb"), filename=associated_file)
        form.add_field("primary_version_id", primary_version_id)

        async with session.post(
            f"{base_url}/add-associated-files/{asset_path}",
            headers={"Authorization": f"Bearer {user_token}"},
            data=form
        ) as resp:
            result = await resp.json()

    logger.info(f"Adding associated files to {asset_path}/{primary_version_id} completed")

    return result


@mcp.tool()
async def file_download(
    ctx: Context[ServerSession, None],
    asset_path: str, 
    version_id: str,
) -> Dict:
    
    """
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
    """
    user_token = get_bearer_token(ctx)

    logger.info(f"Downloading {asset_path}/{version_id} ...")

    # Since the base64 encoded file string may be too long, we don't return it.
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{base_url}/filedownload/{asset_path}/{version_id}?return_file_content=False",
            headers={"Authorization": f"Bearer {user_token}"}
        ) as resp:
            result = await resp.json()

    logger.info(f"Download of {asset_path}/{version_id} completed")

    return result


@mcp.tool()
async def list_file_versions(
    ctx: Context[ServerSession, None],
    asset_path: str, 
    primary_filename: str,
) -> Dict:
    
    """
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
    """
    user_token = get_bearer_token(ctx)

    logger.info(f"Listing versions of {asset_path}/{primary_filename} ...")

    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{base_url}/fileversions/{asset_path}/{primary_filename}",
            headers={"Authorization": f"Bearer {user_token}"}
        ) as resp:
            result = await resp.json()

    logger.info(f"Listing versions of {asset_path}/{primary_filename} completed")

    return {"result": result}


@mcp.tool()
async def file_archive(
    ctx: Context[ServerSession, None],
    asset_path: str, 
    version_id: str,
) -> Dict:
    
    """
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
    """
    user_token = get_bearer_token(ctx)

    logger.info(f"Archiving {asset_path}/{version_id} ...")

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/filearchive/{asset_path}/{version_id}",
            headers={"Authorization": f"Bearer {user_token}"}
        ) as resp:
            result = await resp.json()

    logger.info(f"Archiving {asset_path}/{version_id} completed")

    return result


@mcp.tool()
async def file_delete(
    ctx: Context[ServerSession, None],
    asset_path: str, 
    version_id: str,
) -> Dict:

    """
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
    """
    user_token = get_bearer_token(ctx)

    logger.info(f"Deleting {asset_path}/{version_id} ...")

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/delfile/{asset_path}/{version_id}",
            headers={"Authorization": f"Bearer {user_token}"}
        ) as resp:
            result = await resp.json()

    logger.info(f"Deleting {asset_path}/{version_id} completed")

    return result



if __name__ == "__main__":
    # mcp.run(transport="stdio") # for mcpo
    mcp.run(transport="streamable-http")

