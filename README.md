# Digital Twin Energy — Quick Start

## Prerequisites
- Docker
- Docker Compose

## Run core services
```bash
docker-compose up -d
```

## Access Services
- **Backend API**: http://localhost:8000/healthz
- **pgAdmin**: http://localhost:5050 (login: admin@local / admin)
- **MQTT Broker**: mqtt://localhost:1883 (or ws://localhost:9001 for WebSocket)
- **TimescaleDB**: localhost:5432 (user: energy, password: energy, database: energydb)

## Stop Services
```bash
docker-compose down -v
```

## Project Structure
- `backend_ai/` - FastAPI backend application
- `database/` - Database initialization scripts
- `mqtt_broker/` - MQTT broker configuration

## Development
To run the backend in development mode with hot-reload:
```bash
docker-compose up -d db mqtt  # Start dependencies
cd backend_ai
uvicorn main:app --reload   # Run FastAPI with auto-reload
```

## CLI Utilities

- **CSV Ingestion**: Load readings into the database from a CSV

  ```bash
  python -m backend_ai.data.ingest_csv \
    --csv path/to/readings.csv \
    --ts timestamp_column \
    --machine machine_id_column \
    --power power_kw_column
  ```

  - Required args:
    - `--csv` Path to CSV file.
    - `--ts` Timestamp column name (ISO-8601 preferred; UTC assumed if naive).
    - `--machine` Machine ID column name.
    - `--power` Power (kW) column name.

- **Create Dataset Splits**: Chronological train/val/test JSON per machine

  ```bash
  python -m backend_ai.data.splits \
    --machine MACHINE_001 \
    --val-days 7 \
    --test-days 7
  ```

  - Output: `datasets/splits/MACHINE_001.json` with timestamp boundaries and index ranges.

## License
MIT - See [LICENSE](LICENSE) for more information.
