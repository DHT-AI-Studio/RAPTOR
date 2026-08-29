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

| Field                 | Type           | Required | Description                                     |
| -------------------- | -------------- | -- | --------------------------------------------- |
| `id`                 | `string`       | ✓  | Unique identifier for the policy, e.g. `M1`, `F2`. |
| `name`               | `string`       | ✓  | Human-readable policy name, e.g. "Medical Misinformation". |
| `description`        | `string`       | ✓  | A short description of this policy, explaining what content this category covers. |
| `severity`           | `enum<string>` | ✓  | Risk level, e.g. `low`, `medium`, `high`, `critical`. |
| `decision`           | `enum<string>` | ✓  | Recommended handling, e.g. `allow`, `warn`, `review`, `block`. |
| `criteria`           | `string[]`     | ✓  | Conditions for determining a violation of this policy; each element represents one judgment criterion. |
| `exceptions`         | `string[]`     | ✗  | Exception cases that should not be judged a violation; may be omitted. |
| `examples.violation` | `string[]`     | ✗  | Violation examples (positive examples), helping the model understand what content should be judged a violation. |
| `examples.allowed`   | `string[]`     | ✗  | Allowed examples (negative examples), helping the model understand what content should not be judged a violation. |


---

| Field                 | Type         | Purpose                                                                                                        |
| -------------------- | ---------- | --------------------------------------------------------------------------------------------------------- |
| `id`                 | `string`   | **Unique identifier for the policy**, typically used as a category code, e.g. `M1`, `F3`, `H2`. The guardrail model's classification result also uses this code. |
| `name`               | `string`   | **Policy name**, for human reading, e.g. "Medical Misinformation," "Financial Fraud," "Hate Speech." Mainly used in the admin UI, documentation, or logs — not necessarily included in the prompt. |
| `description`        | `string`   | **A short description of this policy**, explaining what content this category is meant to catch, usually one to two complete sentences. This is typically placed directly into the prompt so the model understands the category's definition. |
| `severity`           | `string`   | **Risk level**, indicating how severe a violation of this policy is, e.g. `low`, `medium`, `high`, `critical`. Can be used in downstream decisions, e.g. reject outright on high risk, only warn on low risk. |
| `decision`           | `string`   | **Recommended handling**, e.g. `block`, `warn`, `review` (manual review), `allow`. This is information used by the Policy Engine — not necessarily included in the prompt. |
| `criteria`           | `string[]` | **Judgment criteria**, listing the conditions that must be met to fall into this category. The model judges whether content is a violation based on these, so this is usually one of the most important parts of the prompt. |
| `exceptions`         | `string[]` | **Exception cases** — even if some criteria are met, content in these situations should not be judged a violation, e.g. academic discussion, news citation, fiction writing. Helps reduce false positives. |
| `examples.violation` | `string[]` | **Violation examples** (positive examples), showing the model what content should be judged as violating this policy — helps improve the model's understanding. |
| `examples.allowed`   | `string[]` | **Allowed examples** (negative examples), showing the model what content, despite being topically related, should not be judged a violation — helps prevent the model from over-blocking. |
