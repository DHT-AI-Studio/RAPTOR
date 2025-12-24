CREATE TABLE IF NOT EXISTS filemeta (
  dirhash   BIGINT NOT NULL,
  name      VARCHAR(1024) NOT NULL,
  directory VARCHAR(1024),
  meta      BYTEA,
  PRIMARY KEY (dirhash, name)
);
