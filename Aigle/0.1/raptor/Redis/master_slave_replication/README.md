# Redis Master-Slave Replication with Docker

This folder provides a simple Docker-based setup for Redis master-slave replication. It includes configuration files and a `docker-compose.yaml` file to easily deploy a Redis master, a Redis slave, and RedisInsight for monitoring.

---

## 📦 Components

1. **Redis Master**
   - Listens on port `6379`
   - Password protected (`dht888888`)
   - Configured with memory limit (`2GB`) and LRU eviction policy
   - Persists data using **AOF only** (`appendonly yes`)

2. **Redis Slave**
   - Listens on port `6380` (mapped from internal `6379`)
   - Replicates the Redis Master
   - Same authentication as the master
   - Persists data using **RDB only** (`appendonly no`)

3. **RedisInsight**
   - Web UI for managing and monitoring Redis instances
   - Accessible at [http://localhost:5540](http://localhost:5540)

---

## 🔧 Setup Instructions

### 1. Start the Services

Use Docker Compose to start all services:

```bash
docker-compose up -d
```

This will start:

- Redis Master (`redis-master`)
- Redis Slave (`redis-slave`)
- RedisInsight (`redisinsight`)

### 2. Access RedisInsight

Open your browser and go to:

🔗 [http://localhost:5540](http://localhost:5540)

Add the Redis Master instance:

- Host: `redis-master`
- Port: `6379`
- Username: *(optional)*
- Password: `dht888888`

You can now monitor both the master and slave nodes.

---

## 🛠️ Configuration Highlights

### Redis Master (`redis.conf`)

- Binds to `0.0.0.0:6379`
- Authentication: `requirepass dht888888`
- Master auth: `masterauth dht888888`
- Memory limit: `2GB`, `allkeys-lru`
- AOF enabled:
  ```properties
  appendonly yes
  appendfilename "appendonly.aof"
  appendfsync everysec
  ```
- RDB snapshotting:
  ```properties
  save 900 1
  save 300 10
  save 60 10000
  stop-writes-on-bgsave-error no
  ```

### Redis Slave (`redis-slave.conf`)

- Binds to `0.0.0.0:6379`
- Replicates the master: `replicaof redis-master 6379`
- Authentication: same as master
- AOF disabled (`appendonly no`)
- Uses RDB for persistence
- Memory limit: `2GB`, `allkeys-lru`

---

## ✅ Testing Replication

You can test the replication by connecting to the Redis master and writing some keys:

```bash
redis-cli -h localhost -p 6379 -a dht888888
```

Set a key:

```bash
127.0.0.1:6379> SET test_key "Hello Redis"
OK
```

Now connect to the slave:

```bash
redis-cli -h localhost -p 6380 -a dht888888
```

Read the key:

```bash
127.0.0.1:6380> GET test_key
"Hello Redis"
```

The key should be available on the slave node.

---

## 🧹 Data Persistence

- The **master** uses **AOF** (`appendonly.aof`) for better durability.
- The **slave** uses **RDB snapshots** (`dump.rdb`) for persistence.
- Both nodes persist data into Docker volumes for persistence across container restarts.

---

## 🚫 Stop or Reset

To stop all services:

```bash
docker-compose down
```

To remove persistent data:

```bash
docker-compose down -v
```

---

## 📝 Notes

- The master enables **AOF**, while the slave disables it in favor of **RDB**.
- This configuration balances performance and durability between master and slave.
- You can scale this setup by adding more slaves if needed.
- Always secure Redis in production with proper firewall rules and access control.

