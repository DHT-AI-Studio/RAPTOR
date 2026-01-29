#!/bin/bash
# logs.sh
# 用途：查看指定服務的日誌，若未指定則列出所有可選服務

SERVICE=$1

if [ -z "$SERVICE" ]; then
    echo "用法: ./logs.sh <service_name>"
    echo ""
    echo "📦 目前運行中的服務（可選）："
    echo "----------------------------------------"
    docker ps --format "table {{.Names}}" | sed '1d' | sort
    echo "----------------------------------------"
    echo "範例: ./logs.sh audio-process"
    exit 1
fi

# 檢查服務是否存在
if ! docker ps -q -f name="$SERVICE" > /dev/null 2>&1; then
    echo "❌ 錯誤：找不到名為 '$SERVICE' 的運行中容器"
    echo "📦 目前運行中的服務："
    docker ps --format "table {{.Names}}" | sed '1d'
    exit 1
fi

echo "🖨️  正在顯示 '$SERVICE' 的即時日誌...（按 Ctrl+C 停止）"
echo ""
docker logs -f "$(docker ps -q -f name=$SERVICE)"