# PostgreSQL on the Pi — Local Event Store (Phase 10)

The Pi keeps recent plate events in a local PostgreSQL database (the "hot" store).
Phase B (next) syncs them to GCP for the permanent archive; Phase C deletes old
*synced* rows when the drive fills. This doc is Phase A: Postgres running and
`run_live.py` writing to it, plus viewing it from your laptop.

The schema auto-creates on first successful `--db` run, and is already image-ready
(`image_path`) and sync-ready (`synced_to_cloud`) so later phases need no migration.

## 1. Install PostgreSQL

```bash
sudo apt install -y postgresql
```

## 2. Create the database and user

**Pick a password and remember it — you'll use the exact same one in step 4.**
(This example uses `MYPASSWORD`; substitute your own.)

```bash
sudo -u postgres psql <<'SQL'
CREATE USER plates WITH PASSWORD 'MYPASSWORD';
CREATE DATABASE plates OWNER plates;
SQL
```

## 3. Install the Python driver

```bash
cd ~/license-plate-project
source .venv312/bin/activate
uv pip install -r requirements-pi.txt      # brings in psycopg2-binary
```

## 4. Set DATABASE_URL — the #1 gotcha

The app reads the connection from the `DATABASE_URL` environment variable. **The
password here must match step 2 exactly**, and the variable must be set **in the
same terminal you run the logger from**. Persist it in `~/.bashrc` so every
terminal has it:

```bash
sed -i '/DATABASE_URL/d' ~/.bashrc     # remove any old/wrong line
echo 'export DATABASE_URL="dbname=plates user=plates password=MYPASSWORD host=localhost"' >> ~/.bashrc
source ~/.bashrc
echo $DATABASE_URL                      # verify it shows your password
```

## 5. Run the live logger writing to Postgres

```bash
python pi_scripts/run_live.py --db --camera-id gate1
#   add --preview for the live window (run from the Pi's DESKTOP terminal, not SSH)
```

**Check the first line — this is the go/no-go:**
- ✅ `Recording to PostgreSQL (dbname=plates ...)` — connected, table created.
- ❌ `Could not connect to PostgreSQL - falling back to CSV only` — the password in
  `DATABASE_URL` doesn't match step 2 (see Troubleshooting). Rows are going to
  `plate_events.csv`, not the database.

Hold a plate up, **take it away**, wait ~10s (it confirms the vehicle left), and a
row commits. One row per vehicle, written the moment it leaves frame.

## 6. Check what got recorded (on the Pi)

```bash
psql "$DATABASE_URL" -c "SELECT id, plate_text, confidence, first_seen, synced_to_cloud FROM plate_events ORDER BY id DESC LIMIT 10;"
```

Live auto-refreshing view in a terminal:
```bash
watch -n 2 'psql "$DATABASE_URL" -c "SELECT id, plate_text, confidence, first_seen, camera_id FROM plate_events ORDER BY id DESC LIMIT 10;"'
```

## 7. View from your laptop (DBeaver)

DBeaver's built-in SSH tunnel connects without exposing Postgres to the network.
Get the Pi's IP first: `hostname -I` (first address).

New Connection → PostgreSQL:
- **Main tab:** Host `localhost`, Port `5432`, Database `plates`, Username `plates`,
  Password `MYPASSWORD`. (Host is `localhost` because it's tunneled.)
- **SSH tab:** check "Use SSH Tunnel", Host = Pi's IP, Port `22`, User `raspberry`,
  Password = your Pi login.

Test Connection → Finish. Open `plate_events` → Data tab → enable auto-refresh
(circular-arrows button) for a live view.

## Troubleshooting

- **`password authentication failed for user "plates"`** or **"falling back to CSV
  only"** → `DATABASE_URL`'s password ≠ the password from step 2, or `DATABASE_URL`
  isn't set in this terminal. Fix step 4 and re-`source ~/.bashrc`. Confirm the
  password directly: `psql "dbname=plates user=plates password=MYPASSWORD host=localhost" -c "select 1;"`
- **`role "raspberry" does not exist`** → `DATABASE_URL` is empty in this terminal,
  so psql tried your OS username. Set it (step 4).
- **`relation "plate_events" does not exist`** → the table isn't created yet; it
  auto-creates on a successful `--db` run (step 5). Or create it manually:
  `psql "$DATABASE_URL" -f db/schema.sql`.
- **Preview: `could not connect to display`** → you're on SSH; drop `--preview` or
  run from the Pi's desktop terminal.

## What's next

- **Phase B — GCP sync:** push rows where `synced_to_cloud = false` up to BigQuery
  (free tier easily covers text), then mark them synced. Needs a GCP project +
  service-account key.
- **Phase C — retention:** when disk usage crosses a limit, delete the oldest rows
  **that are already synced** to reclaim space. Never deletes un-synced data.
