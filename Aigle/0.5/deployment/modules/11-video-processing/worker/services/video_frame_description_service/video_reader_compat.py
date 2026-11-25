# decord on amd64; OpenCV fallback on arm64 (decord has no aarch64 wheel,
# project unmaintained -- see GX10/ARM64 audit). Both call sites in this
# service already use decord in CPU-decode mode (ctx=cpu(0)), so this is an
# equivalent swap, not a perf downgrade, on the arch where it's needed.
try:
    from decord import VideoReader, cpu
except ImportError:
    import cv2

    def cpu(_device_id: int = 0):
        return None

    class _Frame:
        __slots__ = ("_arr",)

        def __init__(self, arr):
            self._arr = arr

        def asnumpy(self):
            return self._arr

    class VideoReader:
        def __init__(self, video_path, ctx=None, num_threads=1):
            self._cap = cv2.VideoCapture(video_path)
            if not self._cap.isOpened():
                raise RuntimeError(f"cannot open video: {video_path}")
            self._len = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._fps = float(self._cap.get(cv2.CAP_PROP_FPS))

        def __len__(self):
            return self._len

        def get_avg_fps(self):
            return self._fps

        def __getitem__(self, idx):
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame_bgr = self._cap.read()
            if not ok:
                raise IndexError(f"frame {idx} out of range")
            return _Frame(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        def __del__(self):
            self._cap.release()
