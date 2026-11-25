# Module 08 — Media Worker

Shared GPU image for modules 09/10/11/12's media-processing workers (audio/image/video/document),
built once from this Dockerfile and consumed by all of them via `image: raptor/media-worker:${RAPTOR_VERSION}`
— they have no `build:` of their own. Also the `BASE_IMAGE` for module 34's main Dockerfile and
module 35's `Dockerfile.voice`.

## GX10 / ARM64 (Grace-Blackwell, aarch64) deployment checklist

The Dockerfile branches on `TARGETARCH` to install arm64-compatible versions of
torch/torchaudio/torchvision, onnxruntime-gpu and decord (see the Dockerfile's own comments) —
these need no action on your part.

**paddlepaddle-gpu is the one exception that needs a manual step, or it silently degrades to
CPU.** PaddlePaddle has no official arm64 GPU build. On a fresh GX10 host, before building this
image:

1. Check `arm64-wheels/` for an existing `paddlepaddle_gpu*.whl` — if one is already there (e.g.
   copied from another GX10 host), skip to step 3. **Do not try to fetch one from GitHub** — it's
   multiple GB, gitignored on purpose (`arm64-wheels/.gitignore`), and this repo's own PR
   checklist bans committing large binaries. If you need to move one between hosts, copy the
   file directly (scp/rsync/internal artifact store), not through git.
2. Otherwise, run `scripts/build-paddlepaddle-gpu-arm64.sh` **on the GX10 host itself** (needs a
   live GPU + sudo; takes about 40 minutes). It's an unofficial, community-sourced recipe — read
   the script's own header comment for provenance/caveats before trusting it blindly. Copy the
   resulting wheel into `arm64-wheels/`.
3. Build the image. If `arm64-wheels/` has no matching wheel, the build falls back to the
   official CPU-only `paddlepaddle` automatically (PaddleOCR still works, just slower) — a
   missing wheel never fails the build, it only means you forgot step 1/2.

A wheel built on one GX10 is reusable on any other GX10 with the same chip (GB10/sm_121) and a
compatible CUDA/driver version — you don't need to rebuild it per host, only per PaddlePaddle/CUDA
version bump.
