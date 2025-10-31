-- Create TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create energy readings table
CREATE TABLE IF NOT EXISTS energy_readings(
  machine_id TEXT NOT NULL,
  ts TIMESTAMPTZ NOT NULL,
  power_kw DOUBLE PRECISION NOT NULL,
  status TEXT NOT NULL
);

-- Convert to hypertable
SELECT create_hypertable('energy_readings', 'ts', if_not_exists=>TRUE);

-- Optional retention policy (30 days)
-- SELECT add_retention_policy('energy_readings', INTERVAL '30 days');
