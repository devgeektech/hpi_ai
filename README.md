# Central AI Monorepo (HMB + Boardroom + HPI)

Greenfield monorepo with **one Central AI Platform** and three product apps.  
**No Docker** — use the existing Python virtualenv + **PostgreSQL**.

## Architecture

```text
HMB (:8001) ──┐
Boardroom (:8002) ──┼──► Central AI (:8000) ──► CV + HPI engines
HPI (:8003) ──┘         PRIME™ / ALIGN™ live only on HPI app
```

## Setup (venv + Postgres)

1. Install **PostgreSQL** locally and ensure the `postgres` user can connect (default password in examples: `postgres`).

2. Create databases:

```powershell
cd "c:\Users\GT49220\OneDrive\Desktop\Sushil-Projects\OCR-Project"
.\scripts\init_postgres.ps1
```

Or in `psql`:

```sql
CREATE DATABASE central_ai;
CREATE DATABASE hmb;
CREATE DATABASE boardroom;
CREATE DATABASE hpi_app;
```

3. Python deps:

```powershell
.\venv\Scripts\Activate.ps1
pip install -e .\packages\shared
pip install -r requirements\apps.txt
pip install -r requirements.txt
```

4. Optional: copy `.env.example` values into your shell (defaults match local Postgres):

```powershell
$env:CENTRAL_DATABASE_URL = "postgresql+psycopg2://postgres:postgres@localhost:5432/central_ai"
$env:HMB_DATABASE_URL = "postgresql+psycopg2://postgres:postgres@localhost:5432/hmb"
$env:BOARDROOM_DATABASE_URL = "postgresql+psycopg2://postgres:postgres@localhost:5432/boardroom"
$env:HPI_DATABASE_URL = "postgresql+psycopg2://postgres:postgres@localhost:5432/hpi_app"
```

Tables are created automatically when each app starts (`Base.metadata.create_all`).

## Run (4 terminals, venv activated)

```powershell
$env:PYTHONPATH = (Get-Location).Path
uvicorn services.central_ai.app.main:app --reload --port 8000
uvicorn apps.hmb.app.main:app --reload --port 8001
uvicorn apps.boardroom.app.main:app --reload --port 8002
uvicorn apps.hpi.app.main:app --reload --port 8003
```

Or use `.\scripts\run_*.ps1`.

## URLs

| Service | URL |
|---------|-----|
| Central AI | http://127.0.0.1:8000/ |
| Central docs | http://127.0.0.1:8000/docs |
| HMB | http://127.0.0.1:8001/ui |
| Boardroom | http://127.0.0.1:8002/docs |
| HPI | http://127.0.0.1:8003/paths |

## Databases

| Service | Env var | Default DB |
|---------|---------|------------|
| Central AI | `CENTRAL_DATABASE_URL` | `central_ai` |
| HMB | `HMB_DATABASE_URL` | `hmb` |
| Boardroom | `BOARDROOM_DATABASE_URL` | `boardroom` |
| HPI | `HPI_DATABASE_URL` | `hpi_app` |

Image uploads still go under `data/uploads/` (filesystem).

## HPI paths

1. `POST /assessments/analyze`
2. `GET /assessments/{id}`
3. **PRIME™** — `POST /prime/sessions`
4. **ALIGN™** — `POST /align/journeys`

## Entry points

| Port | Module |
|------|--------|
| 8000 | `services.central_ai.app.main:app` |
| 8001 | `apps.hmb.app.main:app` |
| 8002 | `apps.boardroom.app.main:app` |
| 8003 | `apps.hpi.app.main:app` |

## Git commits (author: devgeektech)

This repo’s **local** git identity is `devgeektech <development.geektech@gmail.com>`.

To commit without Cursor co-author trailers:

```powershell
.\scripts\commit.ps1 -Message "Describe your change"
```

Optional paths only:

```powershell
.\scripts\commit.ps1 -Message "Fix stream headers" apps/hpi scripts/commit.ps1
```

Push only when you want it on GitHub:

```powershell
git push origin master
```
