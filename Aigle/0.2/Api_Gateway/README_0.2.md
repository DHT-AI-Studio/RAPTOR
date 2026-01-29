# Raptor 2.0 version Installation Guide
**備註：**
- 該資料夾中的程式碼於 2026/01/12 測試完成可運行
- keycloak 完成以下工作
    - 功能完整可運行
    - SMTP 去識別化
    - 測試用的含有個資的 user 帳號已移除
    - keycloak docker container 建立後創建永久 master_admin 帳號 bash 腳本完成 docker container 掛載＋測試
- 該資料夾下尚未成工作
    - ./keycloak/app/config/email.json 內含有公司 SMPT 帳號密碼尚未移除
    - .env 和 .env.example 去識別化後的同步問題
    - ./keycloak/realm-import/dhtsolution-realm.json 如何使用 .env 資料？
- Python 以下模組為舊版本（=後為新版版本號）
    - fastapi==0.121.2
    - uvicorn==0.38.0
    - PyJWT==2.10.1

## Bulid API Gateway with Docker


### 1. 配置 `.env` 參數
`./keycloak/.env`
```bash
KEYCLOAK_URL=<your keycloak url> # usually "http://localhost:8080"

# keycloak 26.x.x version paramater
# In Keycloak version 26.x.x and later, the initial admin account created at startup is a temporary account.
KC_BOOTSTRAP_ADMIN_USERNAME=<admin_username>
KC_BOOTSTRAP_ADMIN_PASSWORD=<admin_password>
PERMANENT_MASTER_ADMIN_USER=<admin_username>
PERMANENT_ADMIN_PASSWORD=<admin_password>

#  keycloak database
MYSQL_DATABASE=keycloak_database
MYSQL_USER=keycloak
MYSQL_PASSWORD=<mysql_password>
MYSQL_ROOT_PASSWORD=<mysql_root_password>
```

### 2. Installation keycloak docker contianer
1. add execute permission:
```bash
chmod +x ./keycloak/create-permanent-admin.sh
```
2. bulid Docker containers:
```bash
# docker compose up -d
docker compose -f ./keycloak/docker-compose.yml up -d
```
3. view running Docker containers:
```bash
docker ps
```
4. Enter the Docker container to create a permanent admin:  
 
Execute an interactive shell inside the running keycloak container.
```bash
sudo docker exec -it keycloak /bin/sh
```
Change to the Keycloak binary directory inside the container.
```bash
cd /opt/keycloak/bin
```
Run the script that creates a permanent Keycloak admin user.
```bash
./create-permanent-admin.sh
```
If you see "✅ Permanent admin setup complete.", mean that permanent admin is created.

5. exit the container shell session and return to the host system.
```bash
exit
```

### 2️⃣ Installation FastAPI docker contianer
```bash
docker compose up -d
```
---

## 不使用 docker 於本地直接使用 python 運行 API Gateway

### Installation of Python dependencies
1. creat conda enviroment
```bash
conda create -n raptor_APIgateway -y
```
```bash
conda activate raptor_APIgateway
```

2. install Python dependencies
```bash
pip install -r requirements.txt
```

### Initialnize Keycloak
1. add execute permission:
```bash
chmod +x ./keycloak/create-permanent-admin.sh
```
2. bulid Docker containers:
```bash
# docker compose up -d
docker compose -f ./keycloak/docker-compose.yml up -d
```
3. view running Docker containers:
```bash
docker ps
```
4. Enter the Docker container to create a permanent admin:  
 
Execute an interactive shell inside the running keycloak container.
```bash
sudo docker exec -it keycloak /bin/sh
```
Change to the Keycloak binary directory inside the container.
```bash
cd /opt/keycloak/bin
```
Run the script that creates a permanent Keycloak admin user.
```bash
./create-permanent-admin.sh
```
If you see "✅ Permanent admin setup complete.", mean that permanent admin is created.

5. exit the container shell session and return to the host system.
```bash
exit
```

### start FastAPI
```bash
uvicorn app.main:app --reload
```

---
### colse conda and remove conda env
```bash
conda deactivate
```
```bash
conda env list
```
```bash
conda env remove -n <envname>
```
