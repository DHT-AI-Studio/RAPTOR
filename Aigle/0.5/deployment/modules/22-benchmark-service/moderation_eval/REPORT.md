# CM-6 moderation model evaluation: report

Results of Module 23's guard models on the CM-6 evaluation sets, run through
Module 22's `guardrail` / `guardrail_image` pipelines. How to rebuild the sets
and run them is in [README.md](README.md). The full numbers, including counts
and the per-category scores by both methods, are in [results.json](results.json).

The target is F1 ≥ 0.80 per category (`BM_MODERATION_TARGET_F1`). Every model
here is a non-China model.

## Sets

| Set | Cases | safe | hate | sexual | violence | sha256 of the set file |
|---|---|---|---|---|---|---|
| `moderation-text-en` | 686 | 337 | 161 | 147 | 91 | `e638ab6fe1b5764ec0a98e04d180aa32569645875d552e85773d5891603a0ed3` |
| `moderation-text-zh-tw` | 686 | 337 | 161 | 147 | 91 | `ba3002fd091d817604dfe7e39f4bfba3249dcda17ec89a4b793f141ae6be1700` |
| `moderation-image` | 2,509 | 1,260 | 160 | 683 | 406 | depends on `--image-root`; see `manifest.json` |

- Category counts overlap: a case can carry more than one label.
- `manifest.json` has sha256 `23d3774aef6f587fee609b3c646e244bbb3bc54c234fc97b39a76e040d6c7fcc`. It lists every case's source id, content sha256 and labels.
- Sources:
  - `mmathys/openai-moderation-api-evaluation` at revision `84e5cf3bcd6acb3dfc70b6760451645872218a3e`;
  - `yiting/UnsafeBench` at revision `9f4560ae90059237eb5eafc6bd8108c78639d180`.
- The zh-TW set was translated by `gemma4:31b` (685 cases) and `mistral-small3.1:24b` (1 case, after a refusal). No case was left out.

## Runs

All times are UTC+8. Text models were served by Ollama on 192.168.157.135. Image models ran in Module 23 on an RTX 4090 on 192.168.157.163: ShieldGemma 2 in 8-bit, Llama Guard 3 Vision in 4-bit. Module 22 and Module 23 were built from this branch, with a throwaway Postgres and Redis.

| Run | Date | Model (Ollama digest, quantization) | Set | Cases | Errors |
|---|---|---|---|---|---|
| `c95f0921` | 2026-11-04 11:35–11:55 | ShieldGemma 2 4B | image | 2,509 | 0 |
| `37415403` | 2026-11-04 11:55–12:11 | Llama Guard 3 11B Vision | image | 2,509 | 0 |
| `c0c5ebd7` | 2026-11-04 14:05–14:08 | llama-guard3:8b (`46f211c3d866`, Q4_K_M) | EN | 686 | 0 |
| `56d2acb3` | 2026-11-04 14:08–14:11 | llama-guard3:8b | zh-TW | 686 | 0 |
| `e55ff6c2` | 2026-11-04 14:21–16:07 | gpt-oss-safeguard:20b (`f2e795d0099c`, MXFP4) | EN | 686 | 1 |
| `c6f70d80` | 2026-11-04 16:08–18:06 | gpt-oss-safeguard:20b | zh-TW | 686 | 1 |
| `ce38e9bb` | 2026-11-04 18:06–20:23 | granite4.1-guardian:8b (`f82c0882cec1`, Q6_K) | EN | 686 | 0 |
| `96378c7f` | 2026-11-09 02:13–04:25 | granite4.1-guardian:8b | zh-TW | 686 | 1 |

An error is a case with no verdict, a timeout; errors are not scored.

## Results

Each cell is precision / recall / **F1**.

**category**: the model names categories (Llama Guard, ShieldGemma 2), and the cell scores whether it named this one, over every case. **detection**: the model only says safe / unsafe (Granite Guardian, gpt-oss-safeguard), or has no policy for the category; the cell scores whether it flagged the case at all, over this category's cases plus the safe cases.

### Text: EN

| Model | Method | Overall | Hate | Sexual | Violence | p50 / p95 |
|---|---|---|---|---|---|---|
| llama-guard3:8b | category | 0.84 / 0.79 / **0.82** | 0.81 / 0.65 / **0.72** | 0.88 / 0.84 / **0.86** | 0.84 / 0.34 / **0.48** | 0.18 / 0.44 s |
| gpt-oss-safeguard:20b | detection | 0.84 / 0.88 / **0.86** | 0.72 / 0.94 / **0.81** | 0.69 / 0.90 / **0.78** | 0.55 / 0.80 / **0.65** | 5.7 / 23.3 s |
| granite4.1-guardian:8b | detection | 0.76 / 0.90 / **0.82** | 0.59 / 0.91 / **0.72** | 0.58 / 0.93 / **0.71** | 0.44 / 0.88 / **0.59** | 11.5 / 18.5 s |

### Text: zh-TW

| Model | Method | Overall | Hate | Sexual | Violence | p50 / p95 |
|---|---|---|---|---|---|---|
| llama-guard3:8b | category | 0.83 / 0.78 / **0.81** | 0.74 / 0.63 / **0.68** | 0.90 / 0.77 / **0.83** | 0.78 / 0.35 / **0.48** | 0.21 / 0.53 s |
| gpt-oss-safeguard:20b | detection | 0.82 / 0.88 / **0.85** | 0.68 / 0.93 / **0.79** | 0.66 / 0.91 / **0.77** | 0.51 / 0.80 / **0.63** | 7.3 / 23.8 s |
| granite4.1-guardian:8b | detection | 0.79 / 0.87 / **0.83** | 0.63 / 0.85 / **0.72** | 0.62 / 0.93 / **0.75** | 0.49 / 0.87 / **0.63** | 11.4 / 16.4 s |

Llama Guard 3 scored by detection, for comparison with the other two:

| Set | Hate | Sexual | Violence |
|---|---|---|---|
| EN | 0.70 / 0.75 / **0.72** | 0.72 / 0.90 / **0.80** | 0.58 / 0.78 / **0.66** |
| zh-TW | 0.69 / 0.74 / **0.71** | 0.71 / 0.88 / **0.79** | 0.57 / 0.79 / **0.66** |

### Images

| Model | Overall | Hate | Sexual | Violence | p50 / p95 |
|---|---|---|---|---|---|
| ShieldGemma 2, threshold 0.5 | 0.88 / 0.46 / **0.60** | 0.13 / 0.07 / **0.10** (detection: no hate policy) | 0.84 / 0.60 / **0.70** | 0.87 / 0.08 / **0.15** | 0.38 / 0.40 s |
| ShieldGemma 2, best threshold 0.05 | | | 0.73 / 0.79 / **0.76** | 0.88 / 0.11 / **0.19** | |
| Llama Guard 3 11B Vision | 0.44 / 0.33 / **0.38** | 0.00 / 0.00 / **0.00** | 0.28 / 0.39 / **0.33** | 0.00 / 0.00 / **0.00** | 0.36 / 0.41 s |

## Findings

1. **Text: sexual meets the target with Llama Guard 3, hate is close, and violence falls short with every model.**
   - Llama Guard 3 names violence for only 34% of violent cases.
   - The safe / unsafe models flag more of them (78–88%), but also many safe cases (precision 0.44–0.55).
   - Choosing a different model does not fix violence.
2. **zh-TW scores within about 0.04 of EN** for every model, so the guards work on Traditional Chinese about as well as on English.
3. **gpt-oss-safeguard and Granite are 30–60× slower than Llama Guard 3** (p50 5.7–11.5 s against 0.2 s). Callers that wait on a short timeout, such as a 2 s fail-open, would time out on most requests with them.
4. **Images: no model meets the target.**
   - ShieldGemma 2 is best on sexual content.
   - It labels most violent images as "dangerous" or safe, so its violence policy rarely fires.
   - It has no hate policy.
   - Llama Guard 3 Vision flags 42% of safe images (as S12) and never names violence or hate.

## Selection (AC 2)

- **Text: `llama-guard3:8b`.**
  - It is the only model that names the category.
  - It is fast enough to run inline.
  - Its overall F1 (0.82 EN, 0.81 zh-TW) is within 0.04 of the best model.
  - It has no threshold to set: its verdict is safe / unsafe plus categories.
- **Images: ShieldGemma 2, threshold 0.05** for sexual content (F1 0.70 → 0.76; recall 0.60 → 0.79, precision 0.84 → 0.73).
  - Module 23's verdict uses 0.5. CM-2 stores the per-policy scores in `moderation_flags`, so the 0.05 cut-off can be applied when filtering, without re-processing.
  - No threshold makes violence or hate usable.

## Fine-tuning (AC 3)

Not done in this change.

- **Text: needed for violence and hate.**
  - The plan is a LoRA adapter for Llama Guard 3 8B, trained through Module 16 on labelled data kept separate from these sets.
  - It is then re-evaluated with these sets and registered through Module 07.
  - This needs the Hugging Face weights of Llama Guard 3 8B (a gated download), since Module 16 does not train GGUF models.
- **Images: not possible with Module 16, which trains text models only.** Image violence and hate stay a known limit, for a follow-up.
