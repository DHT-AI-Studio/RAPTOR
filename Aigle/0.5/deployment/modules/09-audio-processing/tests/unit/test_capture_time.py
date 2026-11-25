"""P13: the audio orchestrator's capture_time.py is the video orchestrator's, verbatim (no shared package
between the worker containers); its behaviour is tested in 11-video-processing/tests/unit/test_capture_time.py.
Here: it is the same file, and a real WAV recorder date (broadcast extension) is read."""
import pathlib
import shutil
import subprocess

import capture_time
import pytest

HERE = pathlib.Path(__file__).resolve()
VIDEO_COPY = HERE.parents[3] / "11-video-processing/worker/services/video_orchestrator_service/capture_time.py"


def test_same_file_as_the_video_orchestrator():
    assert pathlib.Path(capture_time.__file__).read_bytes() == VIDEO_COPY.read_bytes()


@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="no ffmpeg")
def test_real_wav_recorder_date(tmp_path):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "anullsrc=r=8000:cl=mono", "-t", "1",
                    "-write_bext", "1", "-metadata", "origination_date=2015-03-20",
                    "-metadata", "origination_time=12:15:00", str(tmp_path / "a.wav")], check=True)
    assert capture_time.read_capture_time(str(tmp_path / "a.wav")) == ("2015-03-20T12:15:00", "container_creation_time")


@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="no ffmpeg")
def test_real_mp3_release_date_is_ignored(tmp_path):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "anullsrc=r=8000:cl=mono", "-t", "1",
                    "-metadata", "date=2015", str(tmp_path / "a.mp3")], check=True)
    assert capture_time.read_capture_time(str(tmp_path / "a.mp3")) == (None, None)
