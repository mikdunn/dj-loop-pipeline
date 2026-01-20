# Runs Dropbox OAuth helper in offline mode and writes tokens into .env.
# Usage (from repo root):
#   .\tools\run_dropbox_oauth_offline.ps1

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot '.venv\Scripts\python.exe'
$script = Join-Path $repoRoot 'tools\dropbox_oauth_login.py'

if (-not (Test-Path $python)) {
  throw "Python venv not found at: $python`nCreate/activate the venv first."
}
if (-not (Test-Path $script)) {
  throw "OAuth helper script not found at: $script"
}

Push-Location $repoRoot
try {
  & $python $script --offline --write-env
} finally {
  Pop-Location
}
