import magic
from typing import Union, BinaryIO, Optional
import logging
import io
import os
from .models import MediaType


logger = logging.getLogger(__name__)

try:
    _mime_magic = magic.Magic(mime=True)
except Exception as e:
    logger.error(f"Failed to initialize python-magic. Is libmagic installed? Error: {e}")


BUFFER_INITIAL = 2048
BUFFER_MULTIPLIER = 4
BUFFER_MAX = 1024 * 1024  # 1MB
MIME_TO_EXTENSION = {
    # document formats
    "text/plain": "txt",
    "text/csv": "csv",
    "application/pdf": "pdf",
    "application/msword": "doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.ms-excel": "xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.ms-powerpoint": "ppt",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "application/vnd.oasis.opendocument.text": "odt",
    "application/vnd.oasis.opendocument.spreadsheet": "ods",
    "application/vnd.oasis.opendocument.presentation": "odp",
    "application/json": "json",
    "application/xml": "xml",
    "application/x-yaml": "yaml",

    # image formats
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
    "image/bmp": "bmp",
    "image/svg+xml": "svg",
    "image/heic": "heic",

    # audio formats
    "audio/mpeg": "mp3",
    "audio/wav": "wav",
    "audio/ogg": "ogg",
    "audio/webm": "weba",

    # video formats
    "video/mp4": "mp4",
    "video/x-msvideo": "avi",
    "video/webm": "webm",
    "video/quicktime": "mov",
    "video/x-matroska": "mkv",
}

EXTENSION_TO_MIME = {v: k for k, v in MIME_TO_EXTENSION.items()}


def mime_from_filename(filename: str) -> Optional[str]:
    """
    Return the MIME type based on the file extension from the given filename.

    :param filename: The filename to get the MIME type from.
    :return: The MIME type for the given filename, or None if no match is found.
    """
    
    ext = os.path.splitext(filename)[1].lower().lstrip(".")
    return EXTENSION_TO_MIME.get(ext, None)


def guess_media_type(mime_type: Optional[str]) -> MediaType:
    """Guess MediaType based on MIME type"""
    if not mime_type:
        return MediaType.OTHER
    
    mime_base, *subtype = mime_type.split("/", 1)

    if mime_base == "video":
        return MediaType.VIDEO
    elif mime_base == "image":
        return MediaType.IMAGE
    elif mime_base == "audio":
        return MediaType.AUDIO
    elif mime_base == "text" or (mime_base == "application" and subtype and "zip" not in subtype[0]):
        return MediaType.DOCUMENT
    else:
        return MediaType.OTHER
    

def _detect_mime_from_buffer(file: BinaryIO) -> str:
    """
    Helper function to detect MIME type from a BinaryIO stream by reading buffers.
    """
    original_pos = file.tell()
    current_pos = original_pos
    buffer_size = BUFFER_INITIAL
    
    try:
        mime_type = 'application/octet-stream' # default fallback
        
        while buffer_size <= BUFFER_MAX:
            file.seek(current_pos)
            buffer = file.read(buffer_size)
            
            # If no more data can be read, break the loop
            if not buffer:
                break
                
            new_mime_type = _mime_magic.from_buffer(buffer)
            
            if new_mime_type != 'application/octet-stream':
                mime_type = new_mime_type
                break  # Found a specific MIME type
            
            # Still octet-stream, increase buffer size
            if buffer_size < BUFFER_MAX:
                buffer_size *= BUFFER_MULTIPLIER
                logger.debug(f"Still octet-stream, increasing buffer to {buffer_size}")
            else:
                # Reached maximum buffer size, but still octet-stream
                logger.warning(f"File remains 'application/octet-stream' after checking max buffer size ({BUFFER_MAX} bytes).")
                break

        return mime_type
        
    finally:
        file.seek(original_pos)


def detect_file_type(file: Union[str, BinaryIO, bytes], filename: Optional[str] = None) -> dict:
    """
    Detect MIME type from file path, BinaryIO, or raw bytes.
    prioritizing content detection and using filename to correct generic results.
    """
    try:
        mime_type = None

        # 1. Try to get from filename (fastest but not accurate)
        filename_mime = mime_from_filename(filename)
        if filename_mime:
            logger.debug(f"MIME guessed from filename: {filename_mime}")

        # 2. content detection
        if isinstance(file, str):
            # file path
            content_mime = _mime_magic.from_file(file)
        elif isinstance(file, bytes):
            # raw bytes
            file_stream = io.BytesIO(file)
            content_mime = _detect_mime_from_buffer(file_stream)
        elif hasattr(file, "read") and callable(file.read):
            # BinaryIO
            content_mime = _detect_mime_from_buffer(file)
        else:
            raise ValueError("Unsupported file type: must be path (str), bytes, or a stream (BinaryIO).")

        if content_mime:
            # If magic detection result is too generic, and the filename provides a more specific type, adopt the filename result.
            # 'application/octet-stream' and 'text/plain' are often indicators of magic detection failure or unclear content.
            is_generic = content_mime in ('application/octet-stream', 'text/plain')
            if is_generic and filename_mime:
                logger.info(f"Magic result '{content_mime}' too generic, corrected to filename MIME: {filename_mime}")
                mime_type = filename_mime
            else:
                mime_type = content_mime
                logger.debug(f"MIME detected from content: {content_mime}")
                
        elif filename_mime:
            logger.warning(f"Magic detection failed, falling back to filename MIME: {filename_mime}")
            mime_type = filename_mime

        if not mime_type:
            mime_type = 'application/octet-stream'

        # 3. MIME  normalization
        if mime_type in ("audio/mp3", "audio/x-mp3", "audio/x-mpeg"):
            mime_type = "audio/mpeg"

        # 4. Results packaging
        media_type = guess_media_type(mime_type)
        subtype = MIME_TO_EXTENSION.get(mime_type, mime_type.split("/")[-1])
        
        return {
            "mime_type": mime_type, 
            "base_path": f"{media_type.value}/{subtype}"
        }

    except magic.MagicException as e:
        logger.error(f"libmagic error during file detection: {e}", exc_info=True)
        raise ValueError(f"File type detection failed due to magic library error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during file detection: {e}", exc_info=True)
        raise ValueError(f"Failed to detect file type: {str(e)}")
    

