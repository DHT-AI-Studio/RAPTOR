# Redis Cluster with Docker Compose

This project provides a simple and scalable Docker-based setup for running a **Redis Cluster** with 6 nodes (3 masters + 3 replicas), along with RedisInsight for monitoring. It uses Docker volumes for data persistence and automatically creates the Redis cluster after all nodes are ready.

---

## 📦 Components

1. **Redis Nodes (6)**

   - `redis1` to `redis6`, each listening on port `7000` to `7005`
   - All nodes run in cluster mode
   - Password protected (`dht888888`)
   - AOF persistence enabled
2. **RedisInsight**

   - Web UI for managing and monitoring Redis clusters
   - Accessible at [http://localhost:5540](http://localhost:5540)
3. **Cluster Creator**

   - One-time container that runs the Redis cluster creation command:
     ```bash
     redis-cli --cluster create redis1:7000 redis2:7001 ... --cluster-replicas 1
     ```

---

## 🔧 Setup Instructions

### 1. Start the Cluster

Make sure you have `docker` and `docker-compose` installed, then:

```bash
docker-compose up -d
```

This will start:

- 6 Redis nodes in cluster mode
- RedisInsight for visualization
- A one-off container to initialize the Redis Cluster

> ⏱️ The cluster creation may take a few seconds due to health checks and node timeout settings.

---

### 2. Access RedisInsight

Open your browser and go to:

🔗 [http://localhost:5540](http://localhost:5540)

Add a new connection:

- Connection Type: **Redis Cluster**
- Host: `redis1`
- Port: `7000`
- Name: `Redis Cluster`
- Authentication: `dht888888`

You can now monitor the entire Redis Cluster topology and inspect keys, memory usage, performance metrics, etc.

---

## 📁 Project Structure

```
.
├── docker-compose.yaml    # Docker services definition
├── .env                   # IP and redis password definition
└── redis.conf             # Shared Redis configuration for all nodes
```

---

## 🛠️ Redis Configuration Highlights (`redis.conf`)

```properties
cluster-enabled yes
cluster-config-file nodes.conf
cluster-node-timeout 5000
appendonly yes
protected-mode no
```

- **Cluster Mode**: Enabled with a timeout of 5000ms for node failure detection
- **AOF Persistence**: Enabled for durability

---

## 📁 Docker Volumes

Each Redis node has its own named volume to avoid conflicts:

```yaml
volumes:
  redis1:
  redis2:
  redis3:
  redis4:
  redis5:
  redis6:
```

This ensures:

- Data isolation between nodes
- Easy cleanup using `docker-compose down -v`
- No dependency on local file system paths

---

## 🧪 Testing the Cluster

You can test the Redis Cluster by connecting via `redis-cli`:

```bash
redis-cli -c -p 7000 -a dht888888
```

Try setting and retrieving keys:

```bash
127.0.0.1:7000> SET mykey "Hello Cluster"
-> Redirected to slot [15495] located at 127.0.0.1:7002
OK

127.0.0.1:7000> GET mykey
-> Redirected to slot [15495] located at 127.0.0.1:7002
"Hello Cluster"
```

---

## 🚫 Stop or Reset the Cluster

To stop and remove all containers:

```bash
docker-compose down
```

To stop and remove containers **and volumes**:

```bash
docker-compose down -v
```

This removes all persisted Redis data and resets the cluster state.

---

## ✅ Summary

| Feature                 | Status                       |
| ----------------------- | ---------------------------- |
| Redis Cluster (3M + 3S) | ✅ Ready                     |
| Password Protection     | ✅ Enabled                   |
| AOF Persistence         | ✅ Enabled                   |
| RedisInsight Monitoring | ✅ Included                  |
| Easy Cleanup            | ✅`docker-compose down -v` |
