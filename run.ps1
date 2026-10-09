[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8502
)

$ErrorActionPreference = 'Stop'
$taskProjectRoot = $PSScriptRoot
$taskVenvPython = Join-Path $taskProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskVenvPython)) {
    throw 'Run setup.ps1 first to create the project virtual environment.'
}
Push-Location $taskProjectRoot
try {
    $taskServerLogDirectory = Join-Path $taskProjectRoot 'data\logs'
    New-Item -ItemType Directory -Path $taskServerLogDirectory -Force | Out-Null
    $taskServerLogPath = Join-Path $taskServerLogDirectory 'streamlit-server.log'
    & $taskVenvPython -m streamlit run (Join-Path $taskProjectRoot 'app.py') --server.address 127.0.0.1 --server.port $Port --server.headless true --browser.gatherUsageStats false 2>&1 | Tee-Object -FilePath $taskServerLogPath -Append
    if ($LASTEXITCODE -ne 0) { throw 'Streamlit exited with an error.' }
} finally {
    Pop-Location
}
