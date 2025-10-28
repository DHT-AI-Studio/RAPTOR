FROM python:3.11-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY mcp-server .
COPY requirements.txt .

RUN uv venv
RUN uv pip install -r requirements.txt

# for mcpo
# CMD ["uvx", "mcpo", "--port", "8000", "--", "uv", "run", "mcp-server/test_server.py"] 

# CMD ["uv", "run", "mcp-server/test_server.py"]

# CMD ["uv", "run", "mcp-server/test_server_fastmcp.py"]