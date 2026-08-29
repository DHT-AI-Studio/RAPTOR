-- Raptor PostgreSQL initialization
-- Runs automatically on first container startup via docker-entrypoint-initdb.d

-- ==================== Create per-service databases ====================
-- 冪等建立：當 POSTGRES_DB 已是 mlflow 時，postgres 入口程式已自動建立此庫，
-- 直接 CREATE DATABASE 會報 "already exists" 而中斷 init，因此改用條件式建立。
SELECT 'CREATE DATABASE mlflow'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow')\gexec
