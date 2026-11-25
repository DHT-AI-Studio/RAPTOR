# raptor-grpc-client

Generated Python stubs for the Raptor gRPC gateway: `raptor.v1.raptor_pb2` (messages)
and `raptor.v1.raptor_pb2_grpc` (`RaptorServiceStub`, servicer base). They are generated
from `proto/raptor/v1/raptor.proto` at build time; nothing generated is committed.

```bash
pip install raptor-grpc-client
```

```python
import grpc
from raptor.v1 import raptor_pb2, raptor_pb2_grpc

channel = grpc.insecure_channel("raptor-grpc-gateway:8825")   # or grpc.secure_channel(...)
stub = raptor_pb2_grpc.RaptorServiceStub(channel)
reply = stub.Ping(raptor_pb2.PingRequest(message="hi"),
                  metadata=(("authorization", "Bearer <jwt>"),))   # or ("x-api-key", "<key>")
```

`raptor` is a namespace package, so it can sit next to other Raptor stub packages
(for example `raptor.audio.v1` for the separate AudioService on :8044).

## Building

From `deployment/modules/36-grpc-gateway`:

```bash
bash scripts/build_client.sh            # wheel in client/dist/
```

Needs `grpcio-tools` (already in `requirements.txt`) and network access for pip's build
isolation. The version lives in `client/pyproject.toml`; bump it with the contract.
Where the wheel is published (internal PyPI, GitHub Packages) is not decided yet, so
nothing here publishes.
