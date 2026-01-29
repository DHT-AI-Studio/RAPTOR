#!/bin/bash
# Check service status for development environment with 4 GPUs
# 192.168.157.165
echo "🔍 檢查開發環境服務狀態 (4-GPU)..."

# 錯誤處理函數
error_exit() {
    echo "❌ 錯誤: $1"
    exit 1
}


# 檢查 Docker 服務狀態
check_docker_services() {
    echo "🐳 檢查 Docker 服務狀態..."
    if [ ! -f docker-compose.yaml ]; then
        error_exit "未找到 docker-compose.yaml 文件，請確保文件存在！"
    fi

    # 檢查容器狀態
    docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"
    if [ $? -ne 0 ]; then
        error_exit "無法獲取 Docker 服務狀態，請檢查 docker-compose.yaml 配置！"
    fi

    # 檢查是否有容器處於非運行狀態
    local non_running=$(docker compose ps -q | xargs docker inspect --format '{{.Name}} {{.State.Status}}' | grep -v "running")
    if [ -n "$non_running" ]; then
        echo "⚠️ 以下容器未處於運行狀態："
        echo "$non_running"
        return 1
    else
        echo "✅ 所有 Docker 服務正常運行。"
    fi
}

# 檢查資源使用情況
check_resource_usage() {
    echo "📊 檢查容器資源使用情況..."
    docker compose ps -q | xargs docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
    if [ $? -ne 0 ]; then
        echo "⚠️ 無法獲取資源使用情況，請檢查 Docker 狀態！"
        return 1
    fi
}


# 主檢查流程
main() {
    check_docker_services
    check_resource_usage
    echo "✅ 服務檢查完成！"
}

# 執行主流程
main