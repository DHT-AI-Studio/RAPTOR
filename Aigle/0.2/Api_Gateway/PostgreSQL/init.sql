CREATE TABLE IF NOT EXISTS gateway_users (
    username TEXT PRIMARY KEY,
    hashed_password TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
