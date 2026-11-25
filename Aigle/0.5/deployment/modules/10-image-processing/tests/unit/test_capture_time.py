"""P13: the image orchestrator reads when a photo was taken from its EXIF (capture_time.py); Module 25
validates it and records the photo as an observation event at that time."""
import capture_time
from PIL import Image

EXIF_IFD, DATETIME_ORIGINAL, OFFSET_ORIGINAL = 0x8769, 0x9003, 0x9011


def _jpeg(path, taken=None, offset=None):
    img = Image.new("RGB", (8, 8))
    exif = Image.Exif()
    if taken or offset:
        ifd = exif.get_ifd(EXIF_IFD)
        if taken:
            ifd[DATETIME_ORIGINAL] = taken
        if offset:
            ifd[OFFSET_ORIGINAL] = offset
    img.save(path, "JPEG", exif=exif)


def test_datetime_original_with_its_offset(tmp_path):
    _jpeg(tmp_path / "p.jpg", "2015:03:20 12:15:00", "+08:00")
    assert capture_time.read_capture_time(str(tmp_path / "p.jpg")) == ("2015-03-20T12:15:00+08:00",
                                                                      "exif_datetime_original")


def test_datetime_original_without_an_offset_stays_exif_local_time(tmp_path):
    _jpeg(tmp_path / "p.jpg", "2015:03:20 12:15:00")
    assert capture_time.read_capture_time(str(tmp_path / "p.jpg")) == ("2015:03:20 12:15:00", "exif_datetime_original")


def test_no_exif_date(tmp_path):
    _jpeg(tmp_path / "p.jpg")
    assert capture_time.read_capture_time(str(tmp_path / "p.jpg")) == (None, None)


def test_not_an_image_only_loses_the_date(tmp_path):
    (tmp_path / "x.jpg").write_bytes(b"not an image")
    assert capture_time.read_capture_time(str(tmp_path / "x.jpg")) == (None, None)
    assert capture_time.read_capture_time(str(tmp_path / "missing.jpg")) == (None, None)
