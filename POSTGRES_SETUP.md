# PostgreSQL on the Pi — Local Event Store (Phase 10)

The Pi keeps recent plate events in a local PostgreSQL database (the "hot" store).
Phase B (next) syncs them to GCP for the permanent archive; Phase C deletes old
*synced* rows when the drive fills. This doc is just Phase A: get Postgres running
and have `run_live.py` write to it.

The schema is created automatically on first run, and is already image-ready
(`image_path`) and sync-ready (`synced_to_cloud`) so later phases need no migration.

## 1. Install PostgreSQL

```bash
sudo apt install -y postgresql
```

## 2. Create the database and user

```bash
sudo -u postgres psql <<'SQL'
CREATE USER plates WITH PASSWORD 'changeme';
CREATE DATABASE plates OWNER plates;
SQL
```

(Use a real password. This DB is local to the Pi and not exposed to the network.)

## 3. Install the Python driver

If you set the Pi up before this phase, re-run the install to pick up
`psycopg2-binary`:

```bash
source .venv312/bin/activate
uv pip install -r requirements-pi.txt
```

## 4. Point the app at the database

```bash
export DATABASE_URL="dbname=plates user=plates password=changeme host=localhost"
```

To make it permanent, add that line to `~/.bashrc`.

## 5. Run the live logger, writing to Postgres

```bash
python pi_scripts/run_live.py --db --camera-id gate1
```

Each vehicle is inserted into `plate_events`. If the database is ever unreachable
(or an insert fails), it automatically falls back to `plate_events.csv` so a truck
is never lost.

## 6. Check what got recorded

```bash
psql "$DATABASE_URL" -c "SELECT id, plate_text, confidence, first_seen, synced_to_cloud FROM plate_events ORDER BY id DESC LIMIT 10;"
```

You should see your logged plates, each with `synced_to_cloud = f` (false) — they
become `t` once the GCP sync (Phase B) uploads them.

## What's next

- **Phase B — GCP sync:** a job that pushes rows with `synced_to_cloud = false`
  up to BigQuery (free tier easily covers text), then marks them synced. Needs a
  GCP project + service-account key.
- **Phase C — retention:** when disk usage crosses a limit, delete the oldest rows
  **that are already synced** to reclaim space. Never deletes un-synced data.
