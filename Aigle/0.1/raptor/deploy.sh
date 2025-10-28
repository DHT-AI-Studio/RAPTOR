#!/bin/bash
# Deploy script for development environment with 4 GPUs
# 192.168.157.165
echo "🚀 部署到開發環境 (4-GPU)..."

# 可選: 清理上次殘留資源
#echo "🧹 清理上次殘留資源..."
docker compose down --remove-orphans -v || true

# Step 1: 建立 NFS 目錄結構
echo "📂 確保 NFS 目錄存在..."
cd asset_management
#cp .env.example .env
if ! sudo -E bash ensure_nfs_dirs.sh; then
    echo "❌ NFS 目錄建立失敗！請檢查.env的 NFS 伺服器設定或網路連線。"
    exit 1
fi
cd ..

# Step 2: 執行部署
echo "⚙️ 建立 Docker 服務..."
docker compose build --no-cache
echo "🚢 啟動 Qdant 相關服務..."
docker compose up -d qdrant app_base video_search_api image_search_api audio_search_api document_search_api insert_api 
echo "🚢 啟動 Redis Cluster 相關服務..."
docker compose up -d redis1 redis2 redis3 redis4 redis5 redis6 redisinsight redis-cluster-creator
echo "🚢 啟動 AssetManagement 相關服務..."
docker compose up -d app mysql seaweedfs-master1 seaweedfs-master2 seaweedfs-master3 seaweedfs-volume1 seaweedfs-volume2 seaweedfs-volume3 seaweedfs-volume4 seaweedfs-filer seaweedfs-s3 seaweedfs-admin seaweedfs-worker1 seaweedfs-backup prometheus alertmanager node-exporter grafana lakefs lakefs-gc-cron 
echo "🚢 啟動 ModelLifecycle 相關服務..."
docker compose up -d postgres mlflow api 

echo "🚢 啟動 Kafka 相關服務..."
docker compose up -d kafka-gen cntrl1 cntrl2 cntrl3 kafka1 kafka2 kafka3 akhq redis kafdrop	
echo "✅ 部署完成！請確認所有服務均已成功啟動。"