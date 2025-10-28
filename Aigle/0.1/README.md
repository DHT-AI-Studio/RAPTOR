# RAPTOR - Aigle 0.1

This directory contains the source code for **RAPTOR Aigle 0.1**, the first community beta release.

## Directory Structure

```
Aigle/0.1/
├── README.md           # This file
├── requirements.txt    # Dependencies
├── raptor/             # Main package directory
│   ├── AiModelLifecycle/
│   ├── asset_management/ 
│   ├── kafka/        
│   ├── qdrant_search_docker/
│   ├── Redis/
│   ├── .env
│   ├── dcoker-compose.yaml
│   ├── check-services.sh
│   ├── deploy.sh
│   ├── logs.sh
│   ├── DEPLOYMENT.md
│   └── README.md
├── test_file/             # Test files
│   ├── *.wav
│   ├── *.docx
│   ├── *.csv
│   ├── *.pdf
│   ├── *.mp4
│   └── *.jpg
└── docs/             # Version-specific documentation
    └── ...
```

## Installation

```bash
# Clone the repository
git clone https://github.com/DHT-AI-Studio/RAPTOR.git
cd RAPTOR/Aigle/0.1

# Create virtual environment (recommended)
conda create -n VIE python=3.10
conda activate VIE

# Install required dependencies:
python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
python -m pip install "paddleocr[all]"
pip install -r requirements.txt
```

## Development
### Step 1.
```bash
cd raptor
chmod +x check-services.sh deploy.sh logs.sh rollback.sh stop-all.sh tag-backup.sh
./deploy.sh
```
### Step 2.Deploy service
1. Check container status
    ```bash
    ./check-services.sh
    ```
2. Test API connectivity
    ```bash
    # Modellifecycle 服務
    curl -s http://192.168.157.165:8086/docs

    # Assetmanagement
    curl -s http://192.168.157.165:8010/docs
    ```
3. View service logs
    ```bash
    ./logs.sh <service_name>
    ```

## Quick Start
1. Create a new user and assign a new branch to the user
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
2. Create a new access token for the user
    ```bash
    curl -X 'POST' \
    'http://192.168.157.165:8086/token' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/x-www-form-urlencoded' \
    -d 'grant_type=password&username=user1&password=dht888888&scope=&client_id=string&client_secret=********'
    ```
3. Access RedisInsight
🔗 [http://192.168.157.165:5540](http://192.168.157.165:5540)

    Add a new connection:

    - Connection Type: **Redis Cluster**
    - Host: `redis1`
    - Port: `7000`
    - Name: `Redis Cluster`
    - Authentication: `dht888888`
4. Check if the local Ollama model includes qwen2.5:7b
    ```bash
    ollama list
    ```  
    If not present
    ```bash
    ollama pull qwen2.5:7b
    ```

5. Register MLflow with local Ollama
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
6. Check if the registration was successful
    ```bash
    curl -X 'GET' \
      'http://192.168.157.165:8010/models/registered_in_mlflow?show_all=false' \
      -H 'accept: application/json'
    ```

7. Start audio/video/image/document service  
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
8. Produce a test requests topic
    ```bash
    cd /path/to/kafka/test_service
    python test.py
    ```
9. View Service Logs
    ```bash
    cd path/to/kafka
    tail -f service_name.log
    ```
    Available service names include: `document_orchestrator_service`, `document_analysis_service`, `document_summary_service`, `document_save2qdrant_service`, etc. You can replace `document` with `audio`, `video`, or `image` based on the required service, for example: `audio_orchestrator_service`, `video_analysis_service`, `image_summary_service`, etc
10. Check Redis Data
    ```bash
    sudo docker exec -it redis-kafka_dev redis-cli --raw
    GET "document_orchestrator:correlation_id"
    GET "video_orchestrator:correlation_id"
    GET "audio_orchestrator:correlation_id"
    GET "image_orchestrator:correlation_id"
    ```
    Note: Replace service_name and correlation_id with actual values.
11. Use the Qdrant Search API to query data  
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

## Features

**Multi-Modal Content Processing**: Video, audio, image, and text analysis
- **Semantic Search Engine**: Vector-based similarity search with context understanding
- **AI-Powered Metadata Generation**: Automatic tagging and classification
- **LLM Orchestration Framework**: Flexible integration with multiple language models
- **Content Intelligence Pipeline**: Extract insights from unstructured media
- **Entity Recognition System**: Identify people, places, objects, and concepts
- **Configuration Management**: Flexible configuration for different deployment scenarios
- **Logging and Monitoring**: Comprehensive observability and metrics

## Known Issues

See the main [CHANGELOG.md](../../MAIN_DOCUMENTATION/CHANGELOG.md) for known issues and limitations in this release.

## Documentation

- [Main README](../../README.md)

## Support

- **Issues**: [GitHub Issues](https://github.com/DHT-AI-Studio/RAPTOR/issues)
- **Discussions**: [GitHub Discussions](https://github.com/DHT-AI-Studio/RAPTOR/discussions)
- **Community**: See main README for community channels

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](../../LICENSE) file for details.

## About

RAPTOR Aigle 0.1 is developed by the **DHT Taiwan Team**.

For more information, visit [DHT Solutions](https://dhtsolution.com/).

---

**Version**: Aigle 0.1.0-beta
**Release Date**: October 2025

