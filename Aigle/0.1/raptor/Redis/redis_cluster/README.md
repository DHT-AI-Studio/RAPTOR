# Redis Cluster with Docker Compose

This project provides a robust and scalable Docker-based setup for a **Redis Cluster** (6 nodes: 3 masters + 3 replicas), complete with RedisInsight for monitoring. It supports two distinct deployment modes for flexible networking: **Public Access** and **Internal Network**.

---

## 📁 Project Structure

```
.
├── docker-compose.internal.yml  # Internal Network configuration
├── docker-compose.public.yml    # Public Access configuration
├── README.md
├── redis.conf                   # Shared Redis configuration for all nodes
├── .env.public                # Active variables (Public)
├── .env.internal              # Active variables (Internal)
├── .env.public.example        # Template for Public Mode
└── .env.internal.example      # Template for Internal Mode
```

---

## 📦 Components

1. **Redis Nodes (6)**
   - `redis1` to `redis6`, each listening on port `7000` to `7005`
   - All nodes run in cluster mode
   - Password protected
   - AOF persistence enabled

2. **RedisInsight**
   - Web UI for managing and monitoring Redis clusters
   - Accessible at [http://localhost:5540](http://localhost:5540)

3. **Cluster Creator**

   - A one-off container to automatically initialize the Redis Cluster after all nodes are healthy.

---

## ⚙️ Environment Configuration

You must create the necessary environment files to define the passwords and cluster IP settings before starting. These variables are crucial for Docker Compose to correctly interpret the configuration files.

### 1. Create Environment Files

Copy the example files to create your active configuration files:

```bash
# For Public Access Mode
cp .env.public.example .env.public

# For Internal Network Mode
cp .env.internal.example .env.internal
```

### 2. File Contents & Usage

| File | Content | `IP` Setting Required For... |
| :--- | :--- | :--- |
| **`.env.public`** | `IP=<YOUR_SERVER_IP>`<br>`REDIS_PASSWORD=<YOUR_REDIS_PASSWORD>` | **Redis Cluster Announcement** (for external client routing). |
| **`.env.internal`** | `REDIS_PASSWORD=<YOUR_REDIS_PASSWORD>` | **N/A**. Containers use internal service names for clustering. |

### 3. Setup Notes

  * **Public IP:** Replace `<YOUR_SERVER_IP>` in `.env.public` with the actual IP address that your client applications will use to connect to the Docker host.
  * **Password:** Replace `<YOUR_REDIS_PASSWORD>` in both files with your desired strong password.

---

## 🚀 Starting the Cluster (Choose Your Mode)

Select the mode that fits your deployment needs. Always use the `--env-file` flag to ensure the variables are loaded for YAML substitution.

### Mode A: 🔊 Public Access Mode (External Clients)

This mode exposes all Redis Cluster ports to the host machine, enabling connectivity from outside the Docker network.

**Networking Summary:**

  * **Redis Ports (7000-7005):** **Mapped** to the host.
  * **RedisInsight Port (5540):** **Mapped** to the host.

**💡 Execution Command:**

```bash
docker compose -f docker-compose.public.yml --env-file ./.env.public up -d
```

> **Note:** Ensure ports `7000-7005` and `17000-17005` are open in your host's firewall.

### Mode B: ⚙️ Internal Network Mode (Container-to-Container)

This mode isolates the Redis cluster ports within the Docker network, ideal for applications running as other containers in the same Compose project.

**Networking Summary:**

  * **Redis Ports (7000-7005):** **Not mapped** to the host (internal only).
  * **RedisInsight Port (5540):** **Mapped** to the host for external monitoring access.

**💡 Execution Command:**

```bash
docker compose -f docker-compose.internal.yml --env-file ./.env.internal up -d
```

---

## 📈 Accessing RedisInsight

RedisInsight is accessible in **both modes** via your host machine.

🔗 [http://localhost:5540](http://localhost:5540)

**Connection Details:**

| Field | Public Mode | Internal Mode |
| :--- | :--- | :--- |
| **Host** | The **IP address** you defined in `.env.public` | The Redis service name: `redis1` (Docker internal DNS) |
| **Port** | `7000` | `7000` |
| **Password** | Your defined `REDIS_PASSWORD` | Your defined `REDIS_PASSWORD` |

---

## 🧪 Testing the Cluster

After connecting via RedisInsight, you can verify the cluster's health and functionality directly through the web interface.

### 1\. Check Cluster Topology

1.  In RedisInsight, navigate to the **Browser** or **Analysis Tools** view.
2.  You should see 3 Masters connected and running.

### 2\. Run a Test Command

Use the **CLI** feature within RedisInsight to test key storage and retrieval:

```
# Set a key. The cluster automatically handles redirection.
SET mykey "Hello Cluster from RedisInsight"

# Retrieve the key.
GET mykey
```

You should see an `OK` response for the `SET` command and the stored value for the `GET` command, confirming the cluster is fully operational and key distribution is working.

---

## 🚫 Stopping or Resetting the Cluster

Always use the specific Compose file you used for startup when tearing down the services.

| Operation | Command Example (Public Mode) | Description |
| :--- | :--- | :--- |
| **Stop & Remove Containers** | `docker compose -f docker-compose.public.yml down` | Stops and removes all containers, preserving data volumes. |
| **Stop & Reset Data** | `docker compose -f docker-compose.public.yml down -v` | Stops and removes containers **and** all persisted Redis data volumes. |


