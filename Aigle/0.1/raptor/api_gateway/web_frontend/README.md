# RAPTOR API Gateway Web Frontend

本資料夾提供一個基於 Flask 的示範前端，方便以圖形界面呼叫 RAPTOR API Gateway 的所有端點。

## 功能

- 註冊、登入並管理 JWT Token。
- 透過分頁導覽快速切換搜尋、上傳、資產、處理、聊天等功能。
- 覆蓋影片、音訊、文件、圖片與跨集合的搜尋端點。
- 支援單檔與批次上傳，以及上傳即分析的流程。
- 自動保留最近的資產路徑與版本 ID，可直接套用在資產管理或後續處理表單。
- 完整的資產管理操作：列出版本、下載、封存、刪除。
- 觸發檔案處理、快取查詢與聊天功能。
- 內建健康檢查呼叫。

## 快速開始

1. 建議使用虛擬環境安裝依賴：

```bash
cd Aigle/0.1/raptor/api_gateway/web_frontend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. 啟動開發伺服器：

```bash
python app.py
```

預設會在 `http://192.168.157.165:8013` 提供前端頁面。

3. 首頁導覽列可切換至各項功能分頁；若遠端 API URL 不同，可在「環境設定」分頁更新 Base URL。登入後 Token 會儲存在瀏覽器 Session 中，方便後續操作。
4. 上傳或列出版本後，系統會自動記錄 asset_path / version_id。前往「資產管理」或「資料處理」分頁可使用下拉選單一鍵填入表單，必要時亦可清除歷史紀錄。

> **提醒**：部分端點需要先登入取得 Token 才能呼叫。頁面已標示並在未登入時給予提示。

## 自訂

- 透過環境變數 `RAPTOR_API_BASE_URL` 變更預設 API Base URL。
- 透過 `RAPTOR_WEB_SECRET` 調整 Flask Session 的密鑰。

## 注意事項

- 前端會臨時建立檔案以便重用範例客戶端的上傳邏輯，檔案在請求完成後即會刪除。
- 下載資產時若勾選「同時回傳檔案內容」，頁面會以 Base64 文字顯示原始位元內容，請視需要自行轉存。
