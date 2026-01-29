#!/bin/bash

# 設定顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Checking services status...${NC}"

# 檢查PID檔案是否存在
PID_FILE="service_pids.txt"
if [ ! -f "$PID_FILE" ]; then
    echo -e "${RED}No services running (service_pids.txt not found)${NC}"
    exit 0
fi

# 檢查檔案是否為空
if [ ! -s "$PID_FILE" ]; then
    echo -e "${YELLOW}service_pids.txt is empty${NC}"
    rm -f "$PID_FILE"
    exit 0
fi

echo -e "${YELLOW}Service Status:${NC}"
echo "----------------------------------------"

running_count=0
total_count=0

# 讀取並檢查所有服務
while IFS= read -r pid; do
    pid=$(echo "$pid" | tr -d '[:space:]')
    
    # 檢查PID是否為數字
    if ! [[ "$pid" =~ ^[0-9]+$ ]]; then
        continue
    fi
    
    ((total_count++))
    
    if kill -0 "$pid" 2>/dev/null; then
        echo -e "PID $pid: ${GREEN}RUNNING${NC}"
        ((running_count++))
    else
        echo -e "PID $pid: ${RED}STOPPED${NC}"
    fi
    
done < "$PID_FILE"

echo "----------------------------------------"
echo -e "${YELLOW}Summary: $running_count/$total_count services running${NC}"

if [ $running_count -eq 0 ]; then
    echo -e "${RED}All services are stopped${NC}"
    rm -f "$PID_FILE"
elif [ $running_count -eq $total_count ]; then
    echo -e "${GREEN}All services are running${NC}"
else
    echo -e "${YELLOW}Some services are not running${NC}"
fi
