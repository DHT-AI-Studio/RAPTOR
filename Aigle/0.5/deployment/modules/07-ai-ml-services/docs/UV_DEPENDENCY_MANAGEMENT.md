# UV 依賴管理最佳實踐

## 問題分析

### 為什麼使用 `uv` 還會有版本不一致問題？

即使使用 `uv` 作為快速的 pip 替代品，如果沒有版本鎖定機制，仍會面臨以下問題：

1. **無版本鎖定**
   - `requirements.txt` 只有包名，沒有固定版本
   - 不同時間安裝會得到不同版本
   - 傳遞依賴（依賴的依賴）版本不固定

2. **環境差異**
   - 本地開發環境：可能是某個時間點安裝的版本組合（正常工作）
   - Docker 構建：使用當時最新版本，可能不兼容
   - CI/CD 環境：又是不同的版本組合

3. **Docker 層快取**
   - 即使更新了 requirements.txt，Docker 可能使用快取的舊層
   - 導致安裝的依賴與預期不符

### 本次問題的具體情況

```
本地環境（正常）：       Docker 環境（失敗）：
datasets 4.3.0          datasets 2.14.4
pyarrow 21.0.0          pyarrow 21.0.0
✅ 兼容                 ❌ 不兼容
```

## 解決方案

### 方案 1：使用 `uv` 的完整項目管理（推薦）

#### 1. 創建 `pyproject.toml`
```toml
[project]
name = "ai-model-lifecycle"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "fastapi",
    "pyarrow>=14.0.0,<18.0.0",
    "datasets>=2.19.0",
    # ... 其他依賴
]
```

#### 2. 生成鎖定文件
```bash
uv lock
```
這會生成 `uv.lock` 文件，記錄所有依賴的精確版本。

#### 3. 從鎖定文件導出 requirements
```bash
uv export --no-hashes --no-dev | grep -v "^-e " > requirements.lock.txt
```

#### 4. 在 Dockerfile 中使用鎖定文件
```dockerfile
COPY requirements.lock.txt ./
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache -r requirements.lock.txt
```

### 方案 2：簡單版本固定（臨時方案）

在 `requirements.txt` 中固定關鍵依賴版本：
```txt
pyarrow>=14.0.0,<18.0.0
datasets>=2.19.0
```

## 為什麼這樣就解決了？

### 1. **確定性構建（Deterministic Builds）**
```
uv.lock → requirements.lock.txt → Docker 鏡像
精確版本記錄 → 可重現安裝 → 與本地一致
```

### 2. **版本兼容性保證**
- `uv lock` 會解析所有依賴的兼容版本
- 確保所有包之間的版本相互兼容
- 一次解析，處處使用

### 3. **避免時間依賴**
```
傳統方式：
時間 T1 安裝 → 版本 A（正常）
時間 T2 安裝 → 版本 B（可能失敗）

使用 lock 文件：
任何時間安裝 → 鎖定的版本（總是相同）
```

## 工作流程

### 開發階段
1. 修改依賴：編輯 `pyproject.toml`
2. 更新鎖定：`uv lock`
3. 同步環境：`uv sync`
4. 導出 requirements：`uv export --no-hashes --no-dev | grep -v "^-e " > requirements.lock.txt`

### CI/CD 階段
```dockerfile
COPY requirements.lock.txt ./
RUN uv pip install --system -r requirements.lock.txt
```

### 其他成員
```bash
# 使用完全相同的版本
uv sync --frozen
```

## 最佳實踐總結

### ✅ 應該做的
1. 使用 `pyproject.toml` 管理項目依賴
2. 提交 `uv.lock` 到版本控制
3. 導出並使用 `requirements.lock.txt` 在 Docker 中
4. 在 Dockerfile 中使用 `--no-cache` 避免快取問題
5. 為關鍵依賴添加版本約束

### ❌ 不應該做的
1. 在 `requirements.txt` 中不指定任何版本
2. 忽略 `uv.lock` 文件
3. 依賴 Docker 層快取來安裝依賴
4. 假設"最新版本"總是兼容
5. 本地和 Docker 使用不同的依賴管理方式

## 驗證

### 檢查版本一致性
```bash
# 本地環境
pip list | grep -E "pyarrow|datasets"

# Docker 環境
docker exec <container> pip list | grep -E "pyarrow|datasets"
```

### 應該看到相同的版本
```
datasets    4.0.0
pyarrow     17.0.0
```

## 相關文件

- `pyproject.toml` - 項目配置和依賴聲明
- `uv.lock` - 完整的依賴樹鎖定文件（版本控制）
- `requirements.lock.txt` - 從 uv.lock 導出的 pip 格式（Docker 使用）
- `requirements.txt` - 原始的簡單依賴列表（保留作為參考）

## 參考資源

- [UV 官方文檔](https://github.com/astral-sh/uv)
- [Python 依賴管理最佳實踐](https://packaging.python.org/)
- [確定性構建指南](https://reproducible-builds.org/)
