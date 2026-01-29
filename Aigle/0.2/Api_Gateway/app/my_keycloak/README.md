# Keycloak


## Core Functionality
- SSO(Singo Sigin-On)
- RBAC(Role-based Access Control)

## Project Structure
```text
.
├── app
│   ├── config
│   │   └── email.json
│   ├── dependencies
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-310.pyc
│   │   │   └── security.cpython-310.pyc
│   │   └── security.py
│   ├── __init__.py
│   ├── main.py
│   └── routers
│       ├── creat_user.py
│       ├── email_system.py
│       ├── __init__.py
│       ├── login.py
│       ├── path.py
├── config
│   └── __init__.py
├── docker-compose.yaml
├── README.md
├── realm-import
│   └── dhtsolution-realm.json
├── requirement.txt
└── scripts
    ├── __init__.py
    ├── keycloak_authentication.py
    ├── keycloak_client.py
    ├── keycloak_master.py
    └── keycloak_user.py
```

```text
.
├── app                             # FastAPI(frontend)
│   ├── config
│   │   └── email.json              # email SMTP confuguration
│   ├── dependencies
│   │   ├── __init__.py
│   │   └── security.py             # API authentication
│   ├── __init__.py
│   ├── main.py                     # FastAPI main function
│   └── routers                     # FastAPI Function Implementation
│       ├── creat_user.py           # administrator function using to creat keycloak user account
│       ├── email_system.py         
│       ├── __init__.py
│       ├── login.py                # user login SSO(Singo Sign-On)
│       ├── path.py                 # testing API path
├── config
│   └── __init__.py
├── dhtsolution-realm.json          # keycloak realm configurate file
├── docker-compose.yaml
├── .env.example
├── README.md
├── requirement.txt                 # python dependency
└── scripts                         # keycloak function(backend)
    ├── __init__.py
    ├── keycloak_authentication.py  # user access API keycloak_authenticate
    ├── keycloak_client.py          # keyclok client
    ├── keycloak_master.py          # keycloak master admin function
    └── keycloak_user.py            # basic user function
```


## Keycloak Install and Initialize

### 1. Install Keycloak from docker(啟動 docker 環境)
1. 進入專案根目錄啟動docker
```bash
docker compose up -d
```
2. 查看 contialer 是否已經完成啟動
```bash
docker ps
```
看見 **NAMES** 欄位下方有 `keycloak` 和 `keycloak-db` 表示 contialer 已經完成啟動
```sql
CONTAINER ID   IMAGE                              COMMAND                  CREATED          STATUS                             PORTS                                                             NAMES
2a9d6c68c4da   quay.io/keycloak/keycloak:26.4.7   "/opt/keycloak/bin/k…"   23 seconds ago   Up 11 seconds (health: starting)   8443/tcp, 0.0.0.0:8080->8080/tcp, [::]:8080->8080/tcp, 9000/tcp   keycloak
fd89cedc8e89   mysql:8.0                          "docker-entrypoint.s…"   26 seconds ago   Up 22 seconds (healthy)            33060/tcp, 0.0.0.0:3308->3306/tcp, [::]:3308->3306/tcp            keycloak-db
```

3. 於瀏覽器中打開 Keycloak
```
http://localhost:8080
```
> 初次建立 Keycloak 的環境時，因為資料庫連線需要時間，建議等待5分鐘後在開啟

4. 停止服務
```bash
docker compose down
```
- 停止所有 container
- 不會刪掉 volume（資料庫資料會保留）

如果想連 volume 一起刪掉：
```bash
docker compose down -v
```

### 2. Creat dhtsolution realm


## 永久帳號建立
## 2. Initialize Keycloak



## FastAPI 啟動
在專案根目錄
```bash
python -m uvicorn app.main:app
```

---
### get master admin token
>   令牌預設1分鐘後會失效
```bash
curl \
  -d "client_id=admin-cli" \
  -d "username=admin" \
  -d "password=admin_password" \
  -d "grant_type=password" \
  "http://localhost:8080/realms/master/protocol/openid-connect/token"
```

```bash
curl --request POST \
  --url http://localhost:8080/realms/master/protocol/openid-connect/token \
  --header "Content-Type: application/x-www-form-urlencoded" \
  --data "client_id=admin-cli" \
  --data "grant_type=password" \
  --data "username=admin" \
  --data "password=YOUR_ADMIN_PASSWORD"
```

```bash
curl --request POST \
  --url http://localhost:8080/realms/master/protocol/openid-connect/token \
  --header "Content-Type: application/x-www-form-urlencoded" \
  --data "client_id=admin-cli" \
  --data "grant_type=password" \
  --data "username=admin" \
  --data "password=admin_password"
```


```bash
curl --request GET "http://localhost:8080/admin/realms/{realm}/components"

```
```bash
curl --request GET http://localhost:8080/admin/realms/raptor/components \
    -d "username=admin" \
    -d "password=admin_password" \
    -d "grant_type=password"
```


```bash
curl \
  -d "client_id=admin-cli" \
  -d "username=admin" \
  -d "password=admin_password" \
  -d "grant_type=password" \
  "http://localhost:8080/admin/realms/raptor/components"
```


# user role
|            | developer | basic user | standard user |  
| ---------- | --------- | ---------- | ------------- |  
| creat user |     ✅    |            |               |  
| fine tune  |     ✅    |            |       ✅      |  
| inference  |     ✅    |     ✅     |       ✅      |  


### keycloak 打包成　python module
1️⃣ 確保 keycloak 是 package

先確認 /keycloak 目錄結構：
```bash
keycloak/
├── app/
│   ├── __init__.py
│   └── dependencies/
│       ├── __init__.py
│       └── security.py
├── __init__.py
└── setup.py
```

其中 setup.py 內容可以像這樣：
```python
# keycloak/setup.py
from setuptools import setup, find_packages

setup(
    name="keycloak_pkg",       # 這個名字可以自由命名
    version="0.1",
    packages=find_packages(),  # 自動找所有 __init__.py
)
```

✅ 注意：每個你想 import 的目錄都要有 __init__.py 才會被 setuptools 認為是 package。

2️⃣ 使用 conda 創建環境（如果還沒）
```bash
conda create -n myenv python=3.11
conda activate myenv
```
3️⃣ 安裝 requirements.txt

假設你的 requirements.txt 在專案根目錄（和 /app、/keycloak 同級）：
```python
pip install -r requirements.txt
```

在 conda 環境下，pip 會安裝到該環境，沒問題。

4️⃣ 安裝 local package (keycloak) 到虛擬環境

進入 /keycloak 目錄：
```bash
cd /full/path/to/keycloak
pip install -e .
```

解釋：
`-e` = editable，意思是「開發模式」，修改 keycloak 目錄裡的程式碼會即時生效，不需要重新安裝。
安裝後，整個 /keycloak 會被加入虛擬環境的 site-packages，可以直接 import。

5️⃣ 在 /app/main.py 使用
```python
from keycloak.app.dependencies import security

# 使用裡面的函數
print(security.SOME_CONSTANT)
```

6️⃣ requirements.txt 裡加 local package（可選）

如果想用 requirements.txt 安裝所有東西，包括 keycloak：
```bash
-r requirements.txt       # 原本的依賴
-e ./keycloak             # 本地 editable 安裝 keycloak
```

然後直接：
```bash
pip install -r requirements.txt
```

這樣任何人 clone 專案，用 conda env create && pip install -r requirements.txt 就能自動把 keycloak 安裝進虛擬環境。

💡 總結：

conda + pip 可共存，不影響

pip install -e ./keycloak 最直接

如果要多人開發或部署，建議把 -e ./keycloak 放到 requirements.txt