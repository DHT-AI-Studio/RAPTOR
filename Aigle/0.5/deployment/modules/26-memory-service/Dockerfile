FROM python:3.11-slim-bookworm

WORKDIR /app

RUN groupadd --gid 10026 memsvc \
    && useradd --uid 10026 --gid memsvc --no-create-home --shell /usr/sbin/nologin memsvc

COPY requirements.txt .
# sentence-transformers pulls in torch with no pin of its own -- PyPI's
# default wheel bundles a CUDA runtime newer than this deployment's host
# driver supports, so torch.cuda.is_available() silently comes back False
# and embedding falls back to CPU. Pin the same torch+cu128 build as
# 08-media-worker/17-hybrid-search instead -- confirmed working with real
# GPU memory usage on this host (09/10/11/12, which share module 08's
# image, via ./deploy.sh --status).
RUN pip install --no-cache-dir "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu128 && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128

# libstdc++6: memvid-sdk's aarch64 .so references libstdc++ symbols (e.g.
# std::stringbuf) but doesn't declare it as a NEEDED dependency -- the
# CMD's LD_PRELOAD below needs a real one available up front. gcc +
# libc6-dev: build outline_atomics_shim.c below (gcc alone has no glibc
# headers -- stdint.h lives in libc6-dev). (Checked every other UND
# symbol in the .so via readelf --dyn-syms: everything besides libstdc++
# is either resolved by libc/libpython, already present by then, or
# WEAK, which glibc leaves unresolved without erroring.) Kept after pip
# install so iterating on this line doesn't bust the slow torch/pip
# layer's cache.
RUN apt-get update && apt-get install -y --no-install-recommends curl gcc libc6-dev libstdc++6 && rm -rf /var/lib/apt/lists/*

# memvid-sdk's aarch64 .so also calls GCC's outline-atomics helper
# __aarch64_cas8_sync but never links it in -- confirmed undefined
# everywhere in this image (libgcc_s, libatomic, python3.11 itself,
# every installed site-package .so), and every memvid-sdk release from
# 2.0.152 to 2.0.160 has the same gap, so there's no real library to
# preload our way out of this one. Build it ourselves: standard ABI
# (gcc/libgcc/config/aarch64/lse.S), a plain atomic compare-and-swap.
COPY outline_atomics_shim.c /tmp/outline_atomics_shim.c
RUN gcc -shared -fPIC -O2 -o /usr/local/lib/liboutline_atomics_shim.so /tmp/outline_atomics_shim.c \
    && rm /tmp/outline_atomics_shim.c

COPY app/ .
RUN chown -R memsvc:memsvc /app
# memvid_storage (MEM_STORAGE_ROOT) is an NFS-backed volume (see
# docker-compose.yml) -- don't pre-create/chown it here. On a fresh, empty
# volume Docker copies the image's directory at that path into it and tries
# to preserve its owner/group; the NFS server rejects that chown
# ("operation not permitted"), failing the whole `docker compose up`.
# init-nfs.sh already chmod 777s this tree, which is enough for memsvc to
# create files under it -- app/core/paths.py locks each user's own
# subdirectory down to 0o700 right after creating it.
USER memsvc

EXPOSE 8026

# aarch64 only: memvid-sdk's aarch64 wheel ships a Rust .so with a
# static-TLS segment (confirmed via readelf -l's PT_TLS); glibc's surplus
# for a static-TLS library loaded via a late dlopen (i.e. Python's
# `import`) is too small, so importing it crashes with "cannot allocate
# memory in static TLS block". GLIBC_TUNABLES=glibc.rtld.optional_static_tls
# doesn't exist on this image's glibc (2.36 -- confirmed absent via grep
# on libc.so.6), so LD_PRELOAD it instead, loaded at process start outside
# the dlopen-time surplus entirely (the shim and libstdc++ go first: the
# .so's own undefined symbols need them already in the global scope).
# Not applied on x86_64: its wheel is a different build (not confirmed to
# have the same STATIC_TLS flag at all), and forcing the same
# early-preload there risks surfacing a *new* problem it's never had, for
# a bug it may not even have -- don't fix what isn't broken.
CMD ["sh", "-c", "if [ \"$(uname -m)\" = aarch64 ]; then exec env LD_PRELOAD=/usr/local/lib/liboutline_atomics_shim.so:libstdc++.so.6:/usr/local/lib/python3.11/site-packages/memvid_sdk/_lib.abi3.so uvicorn main:app --host 0.0.0.0 --port 8026; else exec uvicorn main:app --host 0.0.0.0 --port 8026; fi"]
