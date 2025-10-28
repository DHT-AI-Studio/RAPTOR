## Environment Setup

1. Create and activate the Conda environment:
   ```bash
   conda create -n CIE python=3.10
   conda activate CIE
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start Docker services:
   ```bash
   sudo docker compose up -d
   ```
4. Create Kafka topics:
   ```bash
   chmod +x create_topic.sh
   sudo ./create_topic.sh
   ```
## Starting Services

### On machine 165 – Start video/audio/document/image related services:
   ```bash
    cd services
    chmod +x start_services.sh
    ./start_services.sh
   ```


#### Service Management

Check if all services are still running:
```bash
./check_services.sh
```
Stop and delete all services:
```bash
./stop_services.sh
```


You need to adjust the configuration of each service, specifically the CUDA_VISIBLE_DEVICES setting to prevent OOM (Out of Memory).

## Produce a test requests topic

```bash
cd test_service
python test.py
```

## Check Service Status

### View Service Logs
```bash
tail -f service_name.log
```
Available service names include: `document_orchestrator_service`, `document_analysis_service`, `document_summary_service`, `document_save2qdrant_service`, etc.

### Check Redis Data

```bash
sudo docker exec -it redis-kafka redis-cli --raw
GET "document_orchestrator:correlation_id"
GET "video_orchestrator:correlation_id"
GET "audio_orchestrator:correlation_id"
GET "image_orchestrator:correlation_id"
```

Note: Replace service_name and correlation_id with actual values.

