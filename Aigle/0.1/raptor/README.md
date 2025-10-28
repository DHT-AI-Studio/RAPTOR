# VIE01--Merge-2: Video Insight Engine

## Docker Compose integrate checklist
 - [x] [Redis](#redis)
 - [x] [Asset-Management](#asset_management)
 - [x] [Ai-Model-Lifecycle](#AiModleLifecycle)
 - [x] [Kafka](#kafka)
 - [x] [Qdrant](#qdrant)

## Docker Port config (`host port` : `container port`)
| Service name                   | Host:Container Port(s)                         |
|--------------------------------|------------------------------------------------|
| `redis1`                       | `7000:7000`. `17000:17000`                     |
| `redis2`                       | `7001:7001`. `17001:17001`                     |
| `redis3`                       | `7002:7002`. `17002:17002`                     |
| `redis4`                       | `7003:7003`. `17003:17003`                     |
| `redis5`                       | `7004:7004`. `17004:17004`                     |
| `redis6`                       | `7005:7005`. `17005:17005`                     |
| `redisinsight`                 | `5540:5540`                                    |
| `app(assetmanagement)`         | `8086:8000`                                    |
| `qdrant`                       | `6334:6333`                                    |
| `mysql`                        | `3307:3306`                                    |
| `seaweedfs-master1`            | `9343:9333`, `19343:19333`, `1244:1234`        |
| `seaweedfs-master2`            | `9344:9334`, `19344:19334`, `1245:1235`        |
| `seaweedfs-master3`            | `9345:9335`, `19345:19335`, `1246:1236`        |
| `seaweedfs-volume1`            | `8091:8081`, `18091:18081`, `1247:1237`        |
| `seaweedfs-volume2`            | `8092:8082`, `18092:18082`, `1248:1238`        |
| `seaweedfs-volume3`            | `8093:8083`, `18093:18083`, `1249:1239`        |
| `seaweedfs-volume4`            | `8094:8084`, `18094:18084`, `1250:1240`        |
| `seaweedfs-filer`              | `8898:8888`, `18898:18888`, `1251:1241`        |
| `seaweedfs-s3`                 | `8343:8333`, `18343:18333`, `1225:1242`        |
| `seaweedfs-admin`              | `23656:23646`, `33656:33646`                   |
| `prometheus`                   | `9091:9090`                                    |
| `alertmanager`                 | `9094:9093`                                    |
| `node-exporter`                | `9101:9100`                                    |
| `grafana`                      | `3031:3000`                                    |
| `lakefs`                       | `8011:8000`                                    |
| `postgres`                     | `5433:5432`                                    |
| `mlflow`                       | `5000:5000`                                    |
| `api(AiModelLifecycle)`        | `8010:8010`                                    |
| `kafka1`                       | `19002:19092`                                  |
| `kafka2`                       | `19003:19093`                                  |
| `kafka3`                       | `19004:19094`                                  |
| `akhq`                         | `8280:8080`                                    |
| `redis`                        | `6391:6379`                                    |
| `kafdrop`                      | `9020:9000`                                    |
| `qdrant`                       | `6334:6333`                                    |
| `video_search_api`             | `8821:8811`                                    |
| `audio_search_api`             | `8822:8812`                                    |
| `document_search_api`          | `8823:8813`                                    |
| `image_search_api`             | `8824:8814`                                    |
| `insert_api`                   | `8815:8815`                                    |

## API usages

---
### [Redis](Redis/README.md#-usage-example)
---
### [Qdrant](qdrant_search_docker/README.md)
---
### [Asset_management](asset_management/README.md#-api-endpoints)
---
### [AiModelLifecycle](AiModelLifecycle/README.md#api-端點說明)
---

## Workflow

1. Deploy services (DEPOYMENT.md)
2. Create and activate the Conda environment:
    ```bash
    conda create -n CIE python=3.10
    conda activate CIE
    ```
    Install required dependencies:
    ```bash
    python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
    python -m pip install "paddleocr[all]"
    pip install -r requirements.txt
    ```
3. Create a new user and assign a new branch to the user
    ```bash
    curl -X 'POST' \
    'http://192.168.157.165:8086/users' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "username": "user1",
    "password": "dht888888",
    "password_hash": "",
    "branch": "",
    "permissions": [
        "upload",
        "download",
        "list"
    ]
    }'
    ```
4. Create a new access token for the user
    ```bash
    curl -X 'POST' \
    'http://192.168.157.165:8086/token' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/x-www-form-urlencoded' \
    -d 'grant_type=password&username=user1&password=dht888888&scope=&client_id=string&client_secret=********'
    ```

5. Access RedisInsight
🔗 [http://192.168.157.165:5540](http://192.168.157.165:5540)

    Add a new connection:

    - Connection Type: **Redis Cluster**
    - Host: `redis1`
    - Port: `7000`
    - Name: `Redis Cluster`
    - Authentication: `dht888888`

6. Check if the local Ollama model includes qwen2.5:7b
    ```bash
    ollama list
    ```  
    If not present
    ```bash
    ollama pull qwen2.5:7b
    ```

7. Register MLflow with local Ollama
    ```bash
    curl -X 'POST' \
      'http://192.168.157.165:8010/models/register_ollama' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
      "local_model_name": "qwen2.5:7b",
      "model_params": 7,
      "registered_name": "qwenforsummary",
      "set_priority_to_one": false,
      "stage": "production",
      "task": "text-generation-ollama",
      "version_description": "Register qwen2.5:7b local model"
    }'
    ```
8. Check if the registration was successful
    ```bash
    curl -X 'GET' \
      'http://192.168.157.165:8010/models/registered_in_mlflow?show_all=false' \
      -H 'accept: application/json'
    ```

9. Start audio/video/image/document service  
    9.1 Create Kafka topics:
    ```bash
    cd path/to/kafka
    chmod +x create_topic.sh
    sudo ./create_topic.sh
    ```
    9.2 Starting Services
    ```bash
    cd services
    chmod +x start_services.sh
    ./start_services.sh
    ```
    9.3 Check if all services are still running
    ```bash
    ./check_services.sh
    ```
10. Produce a test requests topic
    ```bash
    cd /path/to/kafka/test_service
    python test.py
    ```
11. View Service Logs
    ```bash
    cd path/to/kafka
    tail -f service_name.log
    ```
    Available service names include: `document_orchestrator_service`, `document_analysis_service`, `document_summary_service`, `document_save2qdrant_service`, etc. You can replace `document` with `audio`, `video`, or `image` based on the required service, for example: `audio_orchestrator_service`, `video_analysis_service`, `image_summary_service`, etc
12. Check Redis Data
    ```bash
    sudo docker exec -it redis-kafka_dev redis-cli --raw
    GET "document_orchestrator:correlation_id"
    GET "video_orchestrator:correlation_id"
    GET "audio_orchestrator:correlation_id"
    GET "image_orchestrator:correlation_id"
    ```
    Note: Replace service_name and correlation_id with actual values.
13. Use the Qdrant Search API to query data  
    (8821/video_search, 8822/audio_search, 8823/document_search, 8824/image_search)
    ```bash
    curl -X POST "http://192.168.157.165:8822/audio_search" \
      -H "Content-Type: application/json" \
      -d '{
        "query_text": "OpenAI",
        "embedding_type": "text",
        "limit": 5
      }'
    ```
## System Data Access Links

Below are the links and descriptions for accessing system data:

1. Cache Data View  
URL: http://192.168.157.165:5540/  
Function: Used to view cache data.

2. Qdrant Data Dashboard  
URL: http://192.168.157.165:6334/dashboard#  
Function: Used to view the Qdrant data dashboard.

3. SeaweedFS Filer File Management  
URL: http://192.168.157.165:8898/buckets/lakefs/data/  
Function: Used to view or download files.

