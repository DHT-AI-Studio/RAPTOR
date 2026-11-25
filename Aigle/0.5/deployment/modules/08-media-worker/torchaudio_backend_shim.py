# arm64 (GX10) only. Appended to the installed torchaudio package's own
# __init__.py (see Dockerfile) rather than shipped as a sitecustomize.py --
# tried that first, but this base image's own /usr/lib/python3.12/
# sitecustomize.py sits earlier on sys.path and shadows anything placed in
# site-packages, confirmed live (`import sitecustomize` resolved to the
# system one, ours never ran). Appending directly into torchaudio's own
# __init__.py runs unconditionally on every `import torchaudio`, from
# anywhere, regardless of sys.path order.
#
# torchaudio>=2.9 removed its whole `_backend` module -- AudioMetaData/info/
# load/list_audio_backends are gone entirely (confirmed by diffing the
# installed wheels against 2.8.0, where they're still present but already
# deprecated, and 2.7.1, where they're not deprecated yet). whisperx==3.3.1
# hard-pins pyannote.audio==3.3.2, whose core/io.py still calls all four for
# real audio loading, so `import whisperx` fails with "AttributeError:
# module 'torchaudio' has no attribute 'AudioMetaData'". pyannote.audio 4.x
# dropped this dependency, but whisperx's pin is exact, not a minimum, so
# bumping it isn't a same-scope fix -- shim the missing four via soundfile
# instead (already installed: librosa's own dependency), matching
# pyannote.audio 3.3.2's exact call signatures.
if "AudioMetaData" not in globals():
    import dataclasses

    import torch as _torch

    _BITS_BY_SUBTYPE = {
        "PCM_S8": 8, "PCM_U8": 8, "PCM_16": 16, "PCM_24": 24, "PCM_32": 32,
        "FLOAT": 32, "DOUBLE": 64,
    }

    @dataclasses.dataclass
    class AudioMetaData:
        sample_rate: int
        num_frames: int
        num_channels: int
        bits_per_sample: int
        encoding: str

    def list_audio_backends():
        return ["soundfile"]

    def info(filepath, backend=None):
        import soundfile as sf

        i = sf.info(filepath)
        return AudioMetaData(
            sample_rate=i.samplerate, num_frames=i.frames, num_channels=i.channels,
            bits_per_sample=_BITS_BY_SUBTYPE.get(i.subtype, 16), encoding=i.subtype,
        )

    def load(filepath, frame_offset=0, num_frames=-1, backend=None, **_kw):
        import soundfile as sf

        data, sample_rate = sf.read(filepath, start=frame_offset, frames=num_frames,
                                     dtype="float32", always_2d=True)
        return _torch.from_numpy(data.T.copy()), sample_rate
