#!/bin/bash
KAFKA_CONTAINER="kafka1-kraft"
KAFKA_SERVERS="kafka1:9092,kafka2:9092,kafka3:9092"
echo "開始創建 topics..."


# 上游各檔案分發處理入口 topics
docker exec $KAFKA_CONTAINER kafka-topics --create --topic video-processing-requests --bootstrap-server $KAFKA_SERVERS --partitions 12 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic video-processing-results --bootstrap-server $KAFKA_SERVERS --partitions 12 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic audio-processing-requests --bootstrap-server $KAFKA_SERVERS --partitions 6 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic document-processing-requests --bootstrap-server $KAFKA_SERVERS --partitions 6 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic image-processing-requests --bootstrap-server $KAFKA_SERVERS --partitions 6 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic audio-processing-results --bootstrap-server $KAFKA_SERVERS --partitions 6 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic document-processing-results --bootstrap-server $KAFKA_SERVERS --partitions 6 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic image-processing-results --bootstrap-server $KAFKA_SERVERS --partitions 6 --replication-factor 3 --if-not-exists

docker exec $KAFKA_CONTAINER kafka-topics --create --topic video-analysis-requests --bootstrap-server $KAFKA_SERVERS --partitions 12 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic video-analysis-results --bootstrap-server $KAFKA_SERVERS --partitions 12 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic audio-analysis-requests --bootstrap-server $KAFKA_SERVERS --partitions 12 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic audio-analysis-results --bootstrap-server $KAFKA_SERVERS --partitions 12 --replication-factor 3 --if-not-exists

# 下游處理 video topics
docker exec $KAFKA_CONTAINER kafka-topics --create --topic video-scene-detection-requests --bootstrap-server $KAFKA_SERVERS --partitions 9 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic video-scene-detection-results --bootstrap-server $KAFKA_SERVERS --partitions 9 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic video-ocr-frame-requests --bootstrap-server $KAFKA_SERVERS --partitions 9 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic video-ocr-frame-results --bootstrap-server $KAFKA_SERVERS --partitions 9 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic video-frame-description-requests --bootstrap-server $KAFKA_SERVERS --partitions 9 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic video-frame-description-results --bootstrap-server $KAFKA_SERVERS --partitions 9 --replication-factor 3 --if-not-exists

# 下游處理 audio topics
docker exec $KAFKA_CONTAINER kafka-topics --create --topic audio-recognizer-requests --bootstrap-server $KAFKA_SERVERS --partitions 6 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic audio-recognizer-results --bootstrap-server $KAFKA_SERVERS --partitions 6 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic audio-classifier-requests --bootstrap-server $KAFKA_SERVERS --partitions 6 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic audio-classifier-results --bootstrap-server $KAFKA_SERVERS --partitions 6 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic audio-diarization-requests --bootstrap-server $KAFKA_SERVERS --partitions 6 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic audio-diarization-results --bootstrap-server $KAFKA_SERVERS --partitions 6 --replication-factor 3 --if-not-exists

# 下游處理 document topics
docker exec $KAFKA_CONTAINER kafka-topics --create --topic document-analysis-requests --bootstrap-server $KAFKA_SERVERS --partitions 6 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic document-analysis-results --bootstrap-server $KAFKA_SERVERS --partitions 6 --replication-factor 3 --if-not-exists


# summary topics
docker exec $KAFKA_CONTAINER kafka-topics --create --topic video-summary-requests --bootstrap-server $KAFKA_SERVERS --partitions 6 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic video-summary-results --bootstrap-server $KAFKA_SERVERS --partitions 6 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic audio-summary-requests --bootstrap-server $KAFKA_SERVERS --partitions 6 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic audio-summary-results --bootstrap-server $KAFKA_SERVERS --partitions 6 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic document-summary-requests --bootstrap-server $KAFKA_SERVERS --partitions 6 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic document-summary-results --bootstrap-server $KAFKA_SERVERS --partitions 6 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic image-description-requests --bootstrap-server $KAFKA_SERVERS --partitions 6 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic image-description-results --bootstrap-server $KAFKA_SERVERS --partitions 6 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic image-ocr-requests --bootstrap-server $KAFKA_SERVERS --partitions 6 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic image-ocr-results --bootstrap-server $KAFKA_SERVERS --partitions 6 --replication-factor 3 --if-not-exists

# save2qdrant topics
docker exec $KAFKA_CONTAINER kafka-topics --create --topic video-results-save-qdrant-requests --bootstrap-server $KAFKA_SERVERS --partitions 6 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic video-results-save-qdrant-results --bootstrap-server $KAFKA_SERVERS --partitions 6 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic audio-results-save-qdrant-requests --bootstrap-server $KAFKA_SERVERS --partitions 6 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic audio-results-save-qdrant-results --bootstrap-server $KAFKA_SERVERS --partitions 6 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic document-results-save-qdrant-requests --bootstrap-server $KAFKA_SERVERS --partitions 6 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic document-results-save-qdrant-results --bootstrap-server $KAFKA_SERVERS --partitions 6 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic image-results-save-qdrant-requests --bootstrap-server $KAFKA_SERVERS --partitions 6 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic image-results-save-qdrant-results --bootstrap-server $KAFKA_SERVERS --partitions 6 --replication-factor 3 --if-not-exists

# DLQ topics
docker exec $KAFKA_CONTAINER kafka-topics --create --topic video-processing-dlq --bootstrap-server $KAFKA_SERVERS --partitions 1 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic audio-processing-dlq --bootstrap-server $KAFKA_SERVERS --partitions 1 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic document-processing-dlq --bootstrap-server $KAFKA_SERVERS --partitions 1 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic image-processing-dlq --bootstrap-server $KAFKA_SERVERS --partitions 1 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic video-scene-detection-dlq --bootstrap-server $KAFKA_SERVERS --partitions 1 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic video-ocr-frame-dlq --bootstrap-server $KAFKA_SERVERS --partitions 1 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic audio-recognizer-dlq --bootstrap-server $KAFKA_SERVERS --partitions 1 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic audio-classifier-dlq --bootstrap-server $KAFKA_SERVERS --partitions 1 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic audio-diarization-dlq --bootstrap-server $KAFKA_SERVERS --partitions 1 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic document-analysis-dlq --bootstrap-server $KAFKA_SERVERS --partitions 1 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic video-summary-dlq --bootstrap-server $KAFKA_SERVERS --partitions 1 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic audio-summary-dlq --bootstrap-server $KAFKA_SERVERS --partitions 1 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic document-summary-dlq --bootstrap-server $KAFKA_SERVERS --partitions 1 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic video-results-save-qdrant-dlq --bootstrap-server $KAFKA_SERVERS --partitions 1 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic audio-results-save-qdrant-dlq --bootstrap-server $KAFKA_SERVERS --partitions 1 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic document-results-save-qdrant-dlq --bootstrap-server $KAFKA_SERVERS --partitions 1 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic image-results-save-qdrant-dlq --bootstrap-server $KAFKA_SERVERS --partitions 1 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic video-analysis-dlq --bootstrap-server $KAFKA_SERVERS --partitions 1 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic audio-analysis-dlq --bootstrap-server $KAFKA_SERVERS --partitions 1 --replication-factor 3 --if-not-exists
docker exec $KAFKA_CONTAINER kafka-topics --create --topic video-frame-description-dlq --bootstrap-server $KAFKA_SERVERS --partitions 1 --replication-factor 3 --if-not-exists



echo "所有 topics 創建完成！"
docker exec $KAFKA_CONTAINER kafka-topics --list --bootstrap-server $KAFKA_SERVERS