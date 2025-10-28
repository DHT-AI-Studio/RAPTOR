#!/bin/bash

# 設定顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Stopping all services...${NC}"

# 檢查PID檔案是否存在
if [ ! -f "service_pids.txt" ]; then
    echo -e "${RED}Error: service_pids.txt not found!${NC}"
    exit 1
fi

# 讀取並停止所有服務
while IFS= read -r pid; do
    if [ -n "$pid" ]; then
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${YELLOW}Stopping process PID: $pid${NC}"
            kill "$pid"
            sleep 1
            
            # 如果進程仍在運行，強制終止
            if kill -0 "$pid" 2>/dev/null; then
                echo -e "${RED}Force killing PID: $pid${NC}"
                kill -9 "$pid"
            fi
        else
            echo -e "${YELLOW}Process PID: $pid already stopped${NC}"
        fi
    fi
done < service_pids.txt

# 清理PID檔案
rm -f service_pids.txt

echo -e "${GREEN}All services have been stopped!${NC}"
