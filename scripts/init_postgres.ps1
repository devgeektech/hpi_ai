# Creates the four Postgres databases used by this monorepo.
# Requires: local PostgreSQL installed, `psql` on PATH.
# Default user/password: postgres / postgres  (edit if yours differ)

$ErrorActionPreference = "Stop"
$env:PGPASSWORD = if ($env:PGPASSWORD) { $env:PGPASSWORD } else { "postgres" }
$user = if ($env:PGUSER) { $env:PGUSER } else { "postgres" }
$hostName = if ($env:PGHOST) { $env:PGHOST } else { "localhost" }
$port = if ($env:PGPORT) { $env:PGPORT } else { "5432" }

$databases = @("central_ai", "hmb", "boardroom", "hpi_app")

foreach ($db in $databases) {
    $exists = psql -h $hostName -p $port -U $user -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='$db'"
    if ($exists -eq "1") {
        Write-Host "Database already exists: $db"
    } else {
        Write-Host "Creating database: $db"
        psql -h $hostName -p $port -U $user -d postgres -c "CREATE DATABASE $db;"
    }
}

Write-Host "Done. Tables are created automatically when each FastAPI app starts."
