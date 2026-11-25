# Memory check: does the gateway stay flat for a transfer over 1 GB?

The requirement is that server memory stays constant for a download over 1 GB, and that no
full-file buffering happens on upload. These two scripts measure that without touching any real
storage:

- `fake_asset.py` is a stand-in for module 04. It serves a generated blob (default 1200 MiB) and reads
  and discards uploads. Nothing is stored.
- `measure.py` samples the gateway process's `VmRSS` from `/proc/<pid>/status` every 50 ms while a
  client downloads, uploads and cancels, and prints the RSS at the start, the maximum, and the maximum
  in each third of every run. Flat thirds mean memory does not grow with the amount transferred.

## Run it

Needs a Redis the gateway can reach (any, e.g. `docker run -d --rm -p 16379:6379 redis:7.2-alpine`)
and the module's requirements installed. From `deployment/modules/36-grpc-gateway`:

```bash
# 1. the fake asset service
python scripts/memory_check/fake_asset.py --port 19000 --size-mib 1200 &

# 2. the gateway, pointed at it (an API key, so no Keycloak is needed)
GRPC_GATEWAY_REDIS_HOST=localhost GRPC_GATEWAY_REDIS_PORT=16379 \
GRPC_GATEWAY_ASSET_MANAGEMENT_URL=http://127.0.0.1:19000 GRPC_GATEWAY_API_KEYS=memtest-key \
GRPC_GATEWAY_GRPC_PORT=18825 GRPC_GATEWAY_METRICS_PORT=19825 GRPC_GATEWAY_KAFKA_BOOTSTRAP_SERVERS= \
python -c "import dev_paths; dev_paths.setup(); import asyncio, server; asyncio.run(server.serve())" &

# 3. the measurement (pid of the gateway process)
python scripts/memory_check/measure.py --pid <gateway pid> --size-mib 1200
```

The exit code is 1 if a transfer is incomplete or a staging file is left behind. It does not fail on a
memory number: what counts as acceptable is for a person to read.

## Result (2026-10-21, local machine, Python 3.10, grpcio 1.68, one gateway process)

| Scenario | Moved | RSS start, max, end (MiB) | Max per third of the run (MiB) |
| --- | --- | --- | --- |
| download, fast consumer | 1200 MiB | 73, 85, 84 | 85 / 85 / 85 |
| 3 concurrent downloads | 3600 MiB | 83, 99, 97 | 98 / 98 / 99 |
| slow consumer (50 ms per chunk), cancelled | 100 MiB | 95, 95, 95 | 95 / 95 / 95 |
| 20 cancelled downloads | 160 MiB | 95, 95, 91 | 95 / 95 / 91 |
| **upload** | 1200 MiB | 91, 106, 106 | 103 / 106 / 106 |
| 10 cancelled uploads | none | 106, 107, 107 | 106 / 107 / 107 |

Three passes of this whole sequence through one process (about 19 GB in total) stayed between about 105
and 117 MiB; it did not keep climbing. The idle gateway is about 73 MiB.

**What this shows:** memory in the gateway does not grow with the size of the file in either direction,
and cancelled transfers leave nothing behind. A 1200 MiB download costs about 12 MiB more than idle; three
at once about 25 MiB; the upload about 15 MiB.

**What it does not show, and one thing it found:**

- **Uploads are staged on disk.** The gateway writes the whole upload to a temporary file and then hands it
  to the asset service in one request: the staging file reached 1200 MiB and was removed afterwards (also after
  cancels). Memory is constant, but the wording "written incrementally ... without buffering the full
  file" is not literally what happens: the file is buffered on disk. Check free space in the temp directory of
  the container for large uploads.
- The asset service is fake, so the memory of module 04 and of the real storage is not measured, and the real
  service's response times are not reproduced.
- It runs the gateway as a local process, not the container, and authenticates with an API key, so the JWT
  verification path is not in the measurement.
- Memory in the deployed gateway (the internal test host) was not measured: its process is not visible from here.
