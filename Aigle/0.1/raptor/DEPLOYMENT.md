#  系統部署與回滾手冊
> 適用環境：`192.168.157.165`(4-gpu)
##  完整操作流程圖
        [開始]
          │
          ▼
    [1. 部署新版本] → ./deploy.sh 
          │
          ▼
    [2. 測試驗證] →  curl 
          │         ./logs.sh
          │
          ▼
    [3. 部署下一版] → 回到 Step 1


##  Step 1：部署新版本

### 開發環境（4-GPU）

```bash
cd /path/to/raptor
chmod +x check-services.sh deploy.sh logs.sh rollback.sh stop-all.sh tag-backup.sh
./deploy.sh
```

## Step 2：測試驗證

### 2.1 檢查容器狀態

```bash
./check-services.sh
```

### 2.2 測試 API 連通性 

```bash
# Modellifecycle 服務
curl -s http://localhost:8086/docs

# Assetmanagement
curl -s http://localhost:8010/docs
```

### 2.3 查看服務日誌（如需除錯）

```bash
./logs.sh <service_name>
```


## Step 3：部署下一版

- 修改程式碼或模型  
- 重新執行 ./deploy.sh  

