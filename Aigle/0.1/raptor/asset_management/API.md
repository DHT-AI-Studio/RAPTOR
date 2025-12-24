# Asset Management Service API Documentation

This service provides comprehensive lifecycle management for assets, including uploading, version control, associated file management, and automated archiving or destruction. It integrates **LakeFS** (Version Control), **SeaweedFS** (Object Storage), **PostgreSQL** (Metadata), and **Qdrant** (Vector Database).

---

## 1. General Information

### 1.1 Authentication & Context
Authentication is handled by an API Gateway. The Gateway must inject these mandatory headers:

| Header | Description | Example |
| :--- | :--- | :--- |
| `X-User-ID` | Unique ID of the user. | `550e8400-e29b-41d4-a716-446655440000` |
| `X-Branch-ID` | The storage namespace identifier. | `group1` or `user-uuid` |

### 1.2 Workspace & Branch Isolation
`X-Branch-ID` determines the storage environment:
*   **Shared Workspace:** `X-Branch-ID` = Group ID (e.g., `group1`). Users share the same space.
*   **Private Workspace:** `X-Branch-ID` and `X-User-ID` are both set to the **User's UUID**. Space is isolated to that user.

### 1.3 Asset Lifecycle Status
- **`active`**: Asset is live and accessible.
- **`archived`**: Read-only state, pending destruction.
- **`destroyed`**: Metadata purged and physical files deleted.

### 1.4 Date & TTL Formats
Fields ending in **`_date_or_ttl`** support:
- Relative TTL: e.g., **`30d`** (days), **`2w`** (weeks), **`6m`** (months), **`1y`** (years).
- Absolute Date: **`YYYY-MM-DD`** or **`YYYY/MM/DD`**.
- ISO Datetime: **`YYYY-MM-DDTHH:MM:SS`**.

---

## 2. API Endpoints

### 2.1 Upload Asset

Uploads a primary file and optional associated files.
The system identifies duplicates based on the combination of **File Content (Checksum)** and **Asset Path** within the specific **Branch ID** (Storage Space).

#### Upload Logic Matrix

| Scenario | Content (Checksum) | Asset Path | Branch DB Status (Latest) | System Action | Qdrant/Analysis Handling |
| --- | --- | --- | --- | --- | --- |
| **1. New Upload** | New in Branch | New Path | No Record | **Upload & Analyze** | Create new **Active** Point |
| **2. Version Update** | **Different** | Same Path | `active` or `archived` | **Upload & Analyze (New Version)** | Create new **Active** Points |
| **3. Global Active Hit** | **Same** in Branch | Any Path | `active` | **Merge/Return Info** | No Change (Return active metadata) |
| **4. Archive Lock** | **Same** in Branch | Same Path | **`archived`** | **Reject (400 Error)** | Forbidden (Cannot reactivate archived records, Integrity protection) |
| **5. Cross-Path Reuse** | **Same** in Branch | **Different Path** | **`archived`** | **Upload & Reuse** | **Clone Point**: Reuse vectors/analysis for new **Active** Point |

> **Business Rule**: Within a specific **Branch**, if an asset at the asset path is already `archived` and the uploaded content is identical, the operation is rejected. This ensures the integrity of the archival state. To re-use the same path with the same content, the archived record must be explicitly deleted first.



- **URL**: `POST /fileupload`
- **Content-Type**: `multipart/form-data`

**cURL Example:**
```bash
curl -X POST "http://<host>:8000/fileupload" \
     -H "X-User-ID: user_01" \
     -H "X-Branch-ID: workspace_abc" \
     -F "primary_file=@test123.txt" \
     -F "associated_files=@test.jpg" \
     -F "archive_date_or_ttl=90d" \
     -F "destroy_date_or_ttl=1y"
```

**Response Example:**
```json
{
  "asset_path": "document/txt/test123",
  "version_id": "5da85b77b4f7c8651f33989ba141bd4c7fa74ec9a5ca5c30a72812de24aea3ec",
  "primary_filename": "test123.txt",
  "associated_filenames": [
    ["test.jpg", "5ae40aed0d2aa3038dcd7d22a2e052a2881ead115f387701a83473b77cf819fe"]
  ],
  "upload_date": "2025-12-31T13:18:05.166625+08:00",
  "archive_date": "2026-03-31T13:18:05.166625+08:00",
  "destroy_date": "2027-03-31T13:18:05.166625+08:00",
  "user": "user_01",
  "branch": "workspace_abc",
  "status": "active",
  "checksum": "31e49720a568e64dd4669e9a08b536bc",
  "existence_info": {
    "exists": false,
    "message": "The primary file is a new file",
    "existing_asset_path": null,
    "existing_version_id": null
  }
}
```

**Response Example (Error - Scenario 4):**

```json
{
  "detail": "Asset at this path within your branch is archived with identical content. Overwriting archived assets is forbidden. Please delete the archived record or change the file content/path."
}
```

---

### 2.2 Add Associated Files
Add extra files to an existing asset version.

- **URL**: `POST /add-associated-files/{asset_path:path}`

**cURL Example:**
```bash
curl -X POST "http://<host>:8000/add-associated-files/document/txt/test123" \
     -H "X-User-ID: user_01" \
     -H "X-Branch-ID: workspace_abc" \
     -F "associated_files=@98042.pdf"
```

**Response Example:**
```json
{
  "asset_path": "document/txt/test123",
  "version_id": "5da85b77b4f7c8651f33989ba141bd4c7fa74ec9a5ca5c30a72812de24aea3ec",
  "primary_filename": "test123.txt",
  "associated_filenames": [
    ["test.jpg", "5ae40aed0d2aa3038dcd7d22a2e052a2881ead115f387701a83473b77cf819fe"],
    ["98042.pdf", "2a9e33108b4047ffa7155aa88467c0355361691bc3e4fdbae0eb5ae751bb5c39"]
  ],
  "upload_date": "2025-12-31T13:18:05.166625+08:00",
  "archive_date": "2026-03-31T13:18:05.166625+08:00",
  "destroy_date": "2027-03-31T13:18:05.166625+08:00",
  "status": "active"
}
```

---

### 2.3 Download/Retrieve Asset
Fetch the metadata and generated presigned URLs for all files in an asset.

- **URL**: `GET /filedownload/{asset_path:path}/{version_id}`
- **Query Params**: `return_file_content` (bool)

**cURL Example:**
```bash
curl -X GET "http://<host>:8000/filedownload/document/txt/test123/5da85b77b4f7c8651f33989ba141bd4c7fa74ec9a5ca5c30a72812de24aea3ec?return_file_content=false" \
     -H "X-User-ID: user_01" \
     -H "X-Branch-ID: workspace_abc"
```

**Response Example:**
```json
{
  "metadata": {
    "asset_path": "document/txt/test123",
    "version_id": "5da85b77b4f7c8651f33989ba141bd4c7fa74ec9a5ca5c30a72812de24aea3ec",
    "primary_filename": "test123.txt",
    "associated_filenames": [
      ["test.jpg", "..."],
      ["98042.pdf", "..."]
    ],
    "status": "active"
  },
  "primary_file": {
    "filename": "test123.txt",
    "content_type": "text/plain",
    "version_id": "5da85b...",
    "url": "http://192.168.157.123:8333/lakefs/data/..."
  },
  "associated_file_1": {
    "filename": "test.jpg",
    "content_type": "image/jpeg",
    "version_id": "...",
    "url": "..."
  },
  "associated_file_2": {
    "filename": "98042.pdf",
    "content_type": "application/pdf",
    "version_id": "...",
    "url": "..."
  }
}
```

---

### 2.4 Archive Asset
Manually transition an asset to `archived` status.

- **URL**: `POST /filearchive/{asset_path:path}/{version_id}`

**cURL Example:**
```bash
curl -X POST "http://<host>:8000/filearchive/document/txt/test123/5da85b...ec" \
     -H "X-User-ID: user_01" \
     -H "X-Branch-ID: workspace_abc"
```

**Response Example:**
```json
{
  "asset_path": "document/txt/test123",
  "version_id": "5da85b77b4f7c8651f33989ba141bd4c7fa74ec9a5ca5c30a72812de24aea3ec",
  "primary_filename": "test123.txt",
  "associated_filenames": [
    ["test.jpg", "5ae40aed0d2aa3038dcd7d22a2e052a2881ead115f387701a83473b77cf819fe"],
    ["98042.pdf", "2a9e33108b4047ffa7155aa88467c0355361691bc3e4fdbae0eb5ae751bb5c39"]
  ],
  "upload_date": "2025-12-31T13:18:05.166625+08:00",
  "archive_date": "2026-03-31T13:18:05.166625+08:00",
  "destroy_date": "2027-03-31T13:18:05.166625+08:00",
  "status": "archived"
}
```

---

### 2.5 Destroy Asset
Permanently delete an asset. Asset must be `archived` first.

- **URL**: `POST /delfile/{asset_path:path}/{version_id}`

**cURL Example:**
```bash
curl -X POST "http://<host>:8000/delfile/document/txt/test123/5da85b...ec" \
     -H "X-User-ID: user_01" \
     -H "X-Branch-ID: workspace_abc"
```

**Response Example:**
```json
{
  "asset_path": "document/txt/test123",
  "version_id": "5da85b77b4f7c8651f33989ba141bd4c7fa74ec9a5ca5c30a72812de24aea3ec",
  "primary_filename": "test123.txt",
  "associated_filenames": [
    ["test.jpg", "5ae40aed0d2aa3038dcd7d22a2e052a2881ead115f387701a83473b77cf819fe"],
    ["98042.pdf", "2a9e33108b4047ffa7155aa88467c0355361691bc3e4fdbae0eb5ae751bb5c39"]
  ],
  "upload_date": "2025-12-31T13:18:05.166625+08:00",
  "archive_date": "2026-03-31T13:18:05.166625+08:00",
  "destroy_date": "2027-03-31T13:18:05.166625+08:00",
  "status": "destroyed"
}
```

---

### 2.6 List File Versions
Retrieve all historical versions for a specific file.

- **URL**: `GET /fileversions/{asset_path:path}/{filename}`

**cURL Example:**
```bash
curl -X GET "http://<host>:8000/fileversions/document/txt/test123/test123.txt" \
     -H "X-User-ID: user_01" \
     -H "X-Branch-ID: workspace_abc"
```

**Response Example:**
```json
[
  {
    "key": "document/txt/test123/test123.txt",
    "version_id": "5da85b77b4f7c8651f33989ba141bd4c7fa74ec9a5ca5c30a72812de24aea3ec",
    "upload_date": "2025-12-31T13:18:05.166625+08:00",
    "status": "archived",
    "url": "http://192.168.157.123:8333/lakefs/data/..."
  },
  ...
]
```

---

### 2.7 Get User Commit History
Get paginated history of assets uploaded/modified by the current user context.

- **URL**: `GET /users/commits`
- **Query Params**: `keyword`, `page`, `page_size`.

**cURL Example:**
```bash
curl -X GET "http://<host>:8000/users/commits?keyword=test&page=1&page_size=5" \
     -H "X-User-ID: user_01" \
     -H "X-Branch-ID: workspace_abc"
```

**Response Example:**
```json
{
  "total_count": 1,
  "total_pages": 1,
  "page": 1,
  "page_size": 5,
  "commits": [
    {
      "asset_path": "document/txt/test123",
      "version_id": "5da85b77b4f7c8651f33989ba141bd4c7fa74ec9a5ca5c30a72812de24aea3ec",
      "primary_filename": "test123.txt",
      "status": "archived"
    },
    ...
  ]
}
```

---

### 2.8 Update Expiration Policies
Modify archiving or destruction dates. If the asset is currently `archived` and the new `archive_date` is set to the future, the asset is automatically reactivated to `active`.

- **URL**: `POST /file-expiration/{asset_path:path}/{version_id}`

**cURL Example:**
```bash
curl -X POST "http://<host>:8000/file-expiration/document/pdf/sample_report/8eeebfa..." \
     -H "X-User-ID: user_01" \
     -H "X-Branch-ID: workspace_abc" \
     -F "archive_date_or_ttl=2026-01-01" \
     -F "destroy_date_or_ttl=2027-09-30"
```

**Response Example:**
```json
{
  "asset_path": "document/pdf/雜糧及蔬菜_PCR_2.0版",
  "version_id": "8eeebfa2c3f412caa5f29c9c92c62664cec2c35aed6b581ae1aa578eee0405c3",
  "primary_filename": "雜糧及蔬菜_PCR_2.0版.pdf",
  "associated_filenames": [
    ["ISO14067_2018.pdf", "3816f48ca391d270e67bb98f5e691cfa8ea1a302c3c4dfd6b01d4e8adcb72f01"],
    ["Raptor_test_20251217.pdf", "3ede9962f2196b723239907c4a959f9818029376373b8567ad5360dba313c3e8"]
  ],
  "upload_date": "2025-12-31T10:37:12.981402+08:00",
  "archive_date": "2026-01-01T00:00:00+08:00",
  "destroy_date": "2027-09-30T00:00:00+08:00",
  "status": "active"
}
```

---

## 3. Automated Lifecycle
The service includes a background scheduler:
1. **Auto-Archive**: Assets exceeding `archive_date` are set to `archived`.
2. **Auto-Destroy**: Assets exceeding `destroy_date` (while archived) are deleted.
3. **Log Cleanup**: Purges Audit Logs older than 120 days daily.
