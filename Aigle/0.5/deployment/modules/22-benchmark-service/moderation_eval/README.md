# Moderation model evaluation (CM-6)

Benchmarks Module 23's guard models on labelled EN, zh-TW and image sets,
through Module 22's `guardrail` / `guardrail_image` pipelines, and reports
precision / recall / F1 per category (`hate`, `sexual`, `violence`).

The sets contain sexual, violent and hateful material, so they are **never
committed or uploaded**. This directory holds only the scripts that rebuild
them from public sources; `manifest.json` (written next to the sets) records
each case's source, source id, sha256 and labels.

**The CM-6 results are in [REPORT.md](REPORT.md)** (numbers in
[results.json](results.json)).

## Sources

| Set | Source | Licence / terms |
|---|---|---|
| `moderation-text-en` | [mmathys/openai-moderation-api-evaluation](https://huggingface.co/datasets/mmathys/openai-moderation-api-evaluation) (OpenAI, *A Holistic Approach to Undesired Content Detection*) | MIT |
| `moderation-text-zh-tw` | the EN set, machine-translated by `translate_zh_tw.py` (`gemma4:31b` on Ollama, `mistral-small3.1:24b` fallback) | as the EN set |
| `moderation-image` | [yiting/UnsafeBench](https://huggingface.co/datasets/yiting/UnsafeBench) (gated: accept its data use agreement on Hugging Face first) | research / education / responsible commercial use |

How labels are derived, and which cases are left out, is in the docstring of
[build_datasets.py](build_datasets.py).

The `--revision` values below are the dataset versions CM-6 was evaluated on;
a later version may hold different cases, and its scores would not compare.
Rebuilding from them gives the same EN and image sets. The zh-TW set is a
machine translation and may come out slightly different when redone.

## Rebuild and run

```bash
SRC=/data/cm6                      # local, not shared
hf download mmathys/openai-moderation-api-evaluation --repo-type dataset --local-dir $SRC/openai-moderation \
    --revision 84e5cf3bcd6acb3dfc70b6760451645872218a3e
hf download yiting/UnsafeBench --repo-type dataset --local-dir $SRC/UnsafeBench \
    --revision 9f4560ae90059237eb5eafc6bd8108c78639d180

python build_datasets.py --src $SRC --out $SRC/schemas \
    --image-root /data/cm6/schemas/images      # where Module 22 sees the images
python translate_zh_tw.py --schemas $SRC/schemas --ollama http://<ollama-host>:11434 \
    --model ollama:gemma4:31b --fallback ollama:mistral-small3.1:24b
#   then have a person check $SRC/schemas/zh-tw-review.csv

python run_eval.py --bench http://localhost:8022/api/v1 --schemas $SRC/schemas --out $SRC/report
```

`run_eval.py` uploads each set as a Module 22 schema and runs every model
(`config_override.model`), one run per model; each run's report is also at
`GET /api/v1/benchmark/runs/{run_id}/moderation` and in MLflow (experiment
`benchmark_<set>`). Module 23 needs all three text guard models active
(`GUARD_MODELS=llama-guard3:8b,granite4.1-guardian:8b,gpt-oss-safeguard:20b`)
and its image guards configured (`SHIELDGEMMA2_MODEL_PATH`,
`LLAMA_GUARD_3_MODEL_PATH`); Module 22 needs the image directory mounted.

## Reading the report

* **category** — the model names categories (Llama Guard, ShieldGemma 2): over
  every case, did it name this one?
* **detection** — the model only says safe / unsafe (Granite Guardian,
  gpt-oss-safeguard), or does not cover the category (ShieldGemma 2 has no
  hate policy): over this category's cases plus the safe cases, did it flag
  the case at all?
* **below target** — categories under `BM_MODERATION_TARGET_F1` (default 0.80),
  the case for fine-tuning.
* **best threshold** — for ShieldGemma 2, the policy threshold with the best F1.
