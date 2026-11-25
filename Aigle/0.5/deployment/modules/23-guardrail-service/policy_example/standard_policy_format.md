# Standard Format for Policy on `RAPTOR`
```json
{
    "id": "string",
    "name": "string",
    "description": "string",
    "severity": "enum(low | medium | high | critical)",
    "decision": "enum(allow | warn | review | block)",
    "criteria": [
        "string"
    ],
    "exceptions": [
        "string"
    ],
    "examples": {
        "violation": [
            "string"
        ],
        "allowed": [
            "string"
        ]
    }
}
```
## Description for all keys and values

| 欄位                   | 資料型態           | 必填 | 說明                                            |
| -------------------- | -------------- | -- | --------------------------------------------- |
| `id`                 | `string`       | ✓  | Policy 的唯一識別碼，例如 `M1`、`F2`。                   |
| `name`               | `string`       | ✓  | Policy 名稱，供人類閱讀，例如「Medical Misinformation」。   |
| `description`        | `string`       | ✓  | 此 Policy 的簡短描述，說明此分類所代表的內容。                   |
| `severity`           | `enum<string>` | ✓  | 風險等級，例如 `low`、`medium`、`high`、`critical`。     |
| `decision`           | `enum<string>` | ✓  | 建議採取的處理方式，例如 `allow`、`warn`、`review`、`block`。 |
| `criteria`           | `string[]`     | ✓  | 判斷是否違反此 Policy 的條件，每個元素代表一項判斷準則。              |
| `exceptions`         | `string[]`     | ✗  | 不應判定為違規的例外情況，可省略。                             |
| `examples.violation` | `string[]`     | ✗  | 違規範例（Positive Examples），協助模型理解哪些內容應判定為違規。     |
| `examples.allowed`   | `string[]`     | ✗  | 合法範例（Negative Examples），協助模型理解哪些內容不應判定為違規。    |


---

| 欄位                   | 型別         | 用途                                                                                                        |
| -------------------- | ---------- | --------------------------------------------------------------------------------------------------------- |
| `id`                 | `string`   | **Policy 的唯一識別碼**，通常用於分類代碼，例如 `M1`、`F3`、`H2`。Guardrail Model 回傳的分類結果也會使用這個代碼。                             |
| `name`               | `string`   | **Policy 名稱**，提供人類閱讀，例如「醫療錯誤資訊」、「金融詐騙」、「仇恨言論」。主要用於管理介面、文件或 Log，不一定需要放進 Prompt。                            |
| `description`        | `string`   | **此 Policy 的簡短描述**，說明此分類要攔截什麼內容，通常是一到兩句完整的描述。這通常會直接放入 Prompt，讓模型了解這個分類的定義。                                |
| `severity`           | `string`   | **風險等級**，表示違反此 Policy 的嚴重程度，例如 `low`、`medium`、`high`、`critical`。可用於後續決策，例如高風險直接拒絕、低風險僅提醒。                 |
| `decision`           | `string`   | **建議採取的處理方式**，例如 `block`（阻擋）、`warn`（警告）、`review`（人工審查）、`allow`（允許）。這是 Policy Engine 使用的資訊，不一定需要放進 Prompt。 |
| `criteria`           | `string[]` | **判斷條件**，列出符合此分類需要滿足的判斷標準。模型會依據這些條件判斷內容是否違規，因此通常是 Prompt 中最重要的部分之一。                                       |
| `exceptions`         | `string[]` | **例外情況**，即使符合部分條件，但屬於這些情境時不應判定違規，例如學術討論、新聞引用、小說創作等，可降低誤判率。                                                |
| `examples.violation` | `string[]` | **違規範例**（Positive Examples），提供模型哪些內容應該判定為違反此 Policy，有助於提升模型理解能力。                                          |
| `examples.allowed`   | `string[]` | **合法範例**（Negative Examples），提供模型哪些內容雖然與主題相關，但不應判定為違規，可避免模型過度攔截。                                           |