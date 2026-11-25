-- RA-9 / RTA-X7 (VIE01-302 / VIE01-327): Module 34 session recordings.
--
-- Like the other service databases this only runs on a brand-new Postgres data volume; on an
-- existing deployment apply it manually: `psql -h $POSTGRES_HOST -U $POSTGRES_USER -f 009_realtime_audio.sql`.
-- The recording consumer also creates the table (IF NOT EXISTS) once the database exists.

CREATE DATABASE realtime_audio; -- 34-realtime-audio

\connect realtime_audio

CREATE TABLE IF NOT EXISTS rt_audio_recordings (
    id          UUID PRIMARY KEY,
    user_id     VARCHAR(255) NOT NULL,
    session_id  VARCHAR(64)  NOT NULL UNIQUE,
    fid         VARCHAR(255) NOT NULL,            -- SeaweedFS file id of the WAV
    duration_s  INTEGER      NOT NULL,
    size_bytes  BIGINT       NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
    expires_at  TIMESTAMPTZ  NOT NULL,            -- created_at + RT_RECORDING_RETENTION_DAYS (90)
    erased_at   TIMESTAMPTZ                       -- GDPR erasure, consent withdrawal or retention
);
CREATE INDEX IF NOT EXISTS idx_rt_audio_recordings_user_id ON rt_audio_recordings (user_id);
CREATE INDEX IF NOT EXISTS idx_rt_audio_recordings_expires_at ON rt_audio_recordings (expires_at) WHERE erased_at IS NULL;
