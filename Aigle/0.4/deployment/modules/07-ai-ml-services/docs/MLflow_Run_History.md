# MLflow Run History

## Description

As a benchmark operator, I want every completed benchmark run logged to MLflow with its scores and metadata, so that I can visualise score improvement curves over time in the MLflow UI - the Raptor equivalent of AutoResearch's nightly experiment trend.

## Acceptance Criteria

- [ ] After each run completes, `run_manager.py` calls `mlflow_logger.log_run(run)`, which creates or reuses an MLflow experiment named `benchmark_{schema_name}`.
- [ ] Each run logs the `aggregate_score` metric and one metric per dimension named `score_{dimension_name}`, such as `score_relevance` and `score_latency`.
- [ ] Each run logs the `schema_id`, `schema_version`, `target_pipeline`, and `test_case_count` tags.
- [ ] The MLflow run URL or run ID is written to `benchmark_runs.mlflow_run_id` in PostgreSQL.
- [ ] The MLflow tracking server URL is configurable with `BM_MLFLOW_URL`, defaulting to `http://raptor-mlflow:5555`.
- [ ] If MLflow is unreachable, the service logs a warning without failing the benchmark run (best effort).
- [ ] `GET /benchmark/schemas/{id}/runs` includes `mlflow_run_id` for each completed run, enabling UI deep links.

## Subtasks

- [ ] Add `MLflowLogger.log_run()` using the MLflow Python SDK in `app/services/mlflow_logger.py`.
- [ ] Call `mlflow_logger.log_run()` at the end of `execute_run()` in `run_manager.py`.
- [ ] Add `BM_MLFLOW_URL` to `app/core/config.py` and `deployment/modules/.env`.
- [ ] Update the `benchmark_runs` PostgreSQL insert to persist `mlflow_run_id`.
- [ ] Add a unit test that mocks `mlflow.set_experiment` and `mlflow.log_metric`, then verifies the expected metrics and tags.