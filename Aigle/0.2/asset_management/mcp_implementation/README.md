# 📦 Asset Management MCP Server

This project wraps the **Asset Management API** as an **MCP Server**, allowing interaction via MCP.  
It provides two server implementations with identical functionality:

- **test_server.py**  
- **test_server_fastmcp.py**  

Both servers expose the same set of tools; the difference is only in the underlying MCP server implementation.

---

## 📂 Project Structure

```

.
├── docker-compose.yaml         # Docker Compose configuration
├── Dockerfile                  # Container build file for MCP Server
├── mcp-server/                 # MCP Server implementations
│   ├── test_server_fastmcp.py  # FastMCP-based MCP Server
│   └── test_server.py          # Low-level MCP Server
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
└── test_mcp.ipynb              # Jupyter Notebook for testing

```

---

## 🐳 Install and Run with Docker Compose

From the project root directory, run:

```bash
docker compose up -d
```

This will:

* Build the Docker images if they don’t exist.
* Start the MCP Server containers in detached mode.


| Service                   | Host Port | Description          |
| ------------------------- | --------- | -------------------- |
| **test\_server\_fastmcp** | `8003`    | FastMCP Server       |
| **test\_server**          | `8004`    | Low-level MCP Server |


Check the status of the services:

```bash
docker compose ps
```

View real-time logs for all services:

```bash
docker compose logs -f
```

View logs for a specific service:

```bash
docker compose logs -f test_server_fastmcp
```

Stop all running containers and remove their associated volumes:

```bash
docker compose down -v
```


⚠️ **Note:** Make sure any files you want the MCP tools to access are mounted into the containers via the `volumes` section of `docker-compose.yaml`.

---

## 🧪 Testing

You can test the MCP Server using **`test_mcp.ipynb`**.
Before running the notebook, set up a Python environment with the required dependencies (same as the Docker container):

```bash
conda create --name mcp --clone base
conda activate mcp
pip install -r requirements.txt
```

Then open `test_mcp.ipynb` and run the test cells.

---

## ⚡ MCP Server Variants

This project provides **two MCP Server implementations** with the **same functionality**, differing only in implementation style:

| Server Type   | Description                                                                                                             |
| ------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **FastMCP**   | High-level wrapper. Faster development, cleaner code. Recommended for most use cases.                                   |
| **Low-level** | Direct low-level server usage. Requires manual handling of sessions and protocol details. Useful for fine-grained control. |

👉 **Recommended:** Use **FastMCP** for new projects unless you specifically need low-level control.

---

## 📁 File Mounting Requirement

⚠️ **Important:** Any files you want to upload **must be placed in a location accessible to the MCP Server**.
This means you need to **mount the host directory into the Docker container via `docker-compose.yaml`**.

Otherwise, the MCP tools will not be able to find or access those files.
For example:

```yaml
services:
  test_server_fastmcp:
    volumes:
      - ./test:/app/test   # Mount local ./test directory into the container
```

---

## Notes

* The servers are designed for demonstration and testing purposes.
* All tools exposed via MCP are identical between the two server implementations.
* Mount your test files into `/app/test` via Docker Compose so the servers can access them.
* ⚠️ **Important:** Ensure that the **Asset Management API** is running and accessible at the configured base URL.  
  The MCP Server wraps this API, so if the API is not available, the MCP tools will not function correctly.
