#!/bin/bash

# 設定顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Activate the environment and start the services...${NC}"

if [ -n "$LD_LIBRARY_PATH" ]; then
    echo -e "${YELLOW}[ENV] System LD_LIBRARY_PATH detected: $LD_LIBRARY_PATH${NC}"
    echo -e "${YELLOW}[ENV] This points to CUDA 12.6 with cuDNN 8.x${NC}"
    echo -e "${YELLOW}[ENV] Clearing to allow PyTorch use bundled cuDNN 9.10.2...${NC}"
    unset LD_LIBRARY_PATH
    echo -e "${GREEN}[ENV] ✓ LD_LIBRARY_PATH cleared${NC}"
else
    echo -e "${GREEN}[ENV] ✓ LD_LIBRARY_PATH not set${NC}"
fi

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

services=(
    # "auth_service/main.py:Auth Service"
    # "asset_management_service/main.py:Asset Management Service"
    "document_orchestrator_service/main.py:Document Orchestrator Service"
    "document_analysis_service/main.py:Document Analysis Service"
    "document_summary_service/main.py:Document Summary Service"
    "document_save2qdrant_service/main.py:Document Save Qdrant Service"
    "image_orchestrator_service/main.py:Image Orchestrator Service"
    "image_processing_service/main.py:Image Processing Service"
    "image_save2qdrant_service/main.py:Image Save Qdrant Service"
    # "audio_orchestrator_service/main.py:Audio Orchestrator Service"
    # "audio_diarization_service/main.py:Audio Diarization Service"
    # "audio_recognizer_service/main.py:Audio Recognizer Service"
    # "audio_classifier_service/main.py:Audio Classifier Service"
    # "audio_summary_service/main.py:Audio Summary Service"
    # "video_orchestrator_service/main.py:Video Orchestrator Service"
    # "video_analysis_service/main.py:Video Analysis Service"
    # "video_scene_detection_service/main.py:Video Scene Detection Service"
    # "video_ocr_frame_service/main.py:Video OCR Frame Service"
)

# Start all services
for service in "${services[@]}"; do
    IFS=':' read -r script_path service_name <<< "$service"
    echo -e "${YELLOW}Start $service_name...${NC}"
    
    # 在背景啟動服務
    env -u LD_LIBRARY_PATH nohup python "$script_path" > /dev/null 2>&1 &
    
    # 儲存PID
    echo $! >> service_pids.txt
    
    sleep 1
done

echo -e "${GREEN}All services have been started!${NC}"
echo -e "${YELLOW}PID saved to service_pids.txt${NC}"
