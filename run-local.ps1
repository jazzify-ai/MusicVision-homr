$ErrorActionPreference = "Stop"

$env:PYTHONUTF8 = "1"
$env:HOMR_PORT = "8010"
$env:HOMR_PRELOAD_MODELS = "true"

& .\.venv\Scripts\uvicorn.exe homr_service.main:app --host 127.0.0.1 --port $env:HOMR_PORT
