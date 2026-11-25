"""When a photo was taken, as its EXIF says (temporal plan P13).

Module 25 validates this and records the photo itself as an observation event at that time ("this photo
was taken on 2015-03-20"); it also anchors deictic phrases in the photo's OCR text.

    EXIF DateTimeOriginal (0x9003)    "YYYY:MM:DD HH:MM:SS", the camera's local clock
    EXIF OffsetTimeOriginal (0x9011)  its UTC offset, when the camera wrote one

Never raises: an image without a readable date is processed exactly as before, with no anchor.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

SOURCE = "exif_datetime_original"
_EXIF_IFD, _DATETIME_ORIGINAL, _OFFSET_ORIGINAL = 0x8769, 0x9003, 0x9011


def read_capture_time(path: str) -> tuple[str | None, str | None]:
    """(the capture time, EXIF form or ISO 8601 with its offset, SOURCE) or (None, None)."""
    try:
        from PIL import Image  # Pillow, already a dependency of this worker
        with Image.open(path) as img:
            ifd = img.getexif().get_ifd(_EXIF_IFD)
        taken = str(ifd.get(_DATETIME_ORIGINAL) or "").strip().strip("\x00")
        offset = str(ifd.get(_OFFSET_ORIGINAL) or "").strip().strip("\x00")
    except Exception as exc:  # noqa: BLE001 -- not an image Pillow reads, damaged EXIF: the file only loses its date; logged
        logger.warning("could not read the EXIF date of %s: %s", path, exc)
        return None, None
    if not taken:
        return None, None
    if offset and len(taken) >= 19:                       # "2015:03:20 12:15:00" + "+08:00" -> ISO with its zone
        return f"{taken[:10].replace(':', '-')}T{taken[11:19]}{offset}", SOURCE
    return taken, SOURCE
