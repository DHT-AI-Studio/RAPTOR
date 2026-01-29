#!/bin/bash

# 設定顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Activate the environment and start the services...${NC}"

# 初始化conda
eval "$(conda shell.bash hook)"

# 激活conda環境
echo -e "${YELLOW}conda activate VIE...${NC}"
conda activate VIE
if [ $? -ne 0 ]; then
    echo -e "${RED}Error：conda activate VIE${NC}"
    exit 1
fi

echo -e "${GREEN}Start services...${NC}"

# 定義服務列表
services=(
    # "auth_service/main.py:Auth Service"
    # "asset_management_service/main.py:Asset Management Service"
    # "document_orchestrator_service/main.py:Document Orchestrator Service"
    # "document_analysis_service/main.py:Document Analysis Service"
    # "document_summary_service/main.py:Document Summary Service"
    "audio_orchestrator_service/main.py:Audio Orchestrator Service"
    "audio_diarization_service/main.py:Audio Diarization Service"
    "audio_recognizer_service/main.py:Audio Recognizer Service"
    "audio_classifier_service/main.py:Audio Classifier Service"
    "audio_summary_service/main.py:Audio Summary Service"
    "audio_save2qdrant_service/main.py:Audio Save Qdrant Service"
    "video_orchestrator_service/main.py:Video Orchestrator Service"
    "video_analysis_service/main.py:Video Analysis Service"
    "audio_analysis_service/main.py:Audio Analysis Service"
    "video_scene_detection_service/main.py:Video Scene Detection Service"
    "video_ocr_frame_service/main.py:Video OCR Frame Service"
    "video_frame_description_service/main.py:Video Frame Description Service"
    "video_summary_service/main.py:Video Summary Service"
    "video_save2qdrant_service/main.py:Video Save Qdrant Service"
)

# 啟動所有服務
for service in "${services[@]}"; do
    IFS=':' read -r script_path service_name <<< "$service"
    echo -e "${YELLOW}Start $service_name...${NC}"
    
    # 在背景啟動服務
    nohup python "$script_path" > /dev/null 2>&1 &
    
    # 儲存PID
    echo $! >> service_pids.txt
    
    sleep 1
done

echo -e "${GREEN}All services have been started!${NC}"
echo -e "${YELLOW}PID saved to service_pids.txt${NC}"
