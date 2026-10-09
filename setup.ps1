[CmdletBinding()]
param(
    [string]$PythonPath,
    [string]$CacheDirectory
)

$ErrorActionPreference = 'Stop'
$taskProjectRoot = $PSScriptRoot
$taskVenvPython = Join-Path $taskProjectRoot '.venv\Scripts\python.exe'
$taskLockFile = Join-Path $taskProjectRoot 'requirements.lock.txt'
$taskUvCommand = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue
$taskUvPath = if ($taskUvCommand) { $taskUvCommand.Source } else {
    Join-Path $env:USERPROFILE '.local\bin\uv.exe'
}
$taskUseUv = Test-Path -LiteralPath $taskUvPath
if (-not (Test-Path -LiteralPath $taskLockFile)) {
    throw 'requirements.lock.txt was not found beside setup.ps1.'
}
if (-not $PythonPath) {
    $taskBundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    if (Test-Path -LiteralPath $taskBundledPython) {
        $PythonPath = $taskBundledPython
    } elseif ($taskUseUv) {
        $PythonPath = '3.12'
    } else {
        $taskPythonCommand = Get-Command python -CommandType Application -ErrorAction SilentlyContinue
        if (-not $taskPythonCommand) { throw 'Install Python 3.12 or pass -PythonPath to its executable.' }
        $PythonPath = $taskPythonCommand.Source
    }
}
if (-not $CacheDirectory) {
    $CacheDirectory = if ($env:UV_CACHE_DIR) { $env:UV_CACHE_DIR } else { Join-Path $taskProjectRoot '.uv-cache' }
}
$taskPreviousCache = $env:UV_CACHE_DIR
$env:UV_CACHE_DIR = $CacheDirectory
Push-Location $taskProjectRoot
try {
    if (-not (Test-Path -LiteralPath $taskVenvPython)) {
        if ($taskUseUv) {
            & $taskUvPath venv --python $PythonPath (Join-Path $taskProjectRoot '.venv')
        } else {
            & $PythonPath -m venv (Join-Path $taskProjectRoot '.venv')
        }
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python virtual environment.' }
    }
    & $taskVenvPython -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)'
    if ($LASTEXITCODE -ne 0) { throw 'This project requires a Python 3.12 virtual environment.' }
    if ($taskUseUv) {
        & $taskUvPath pip install --python $taskVenvPython --requirement $taskLockFile
    } else {
        & $taskVenvPython -m pip install --requirement $taskLockFile
    }
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    $taskEnvFile = Join-Path $taskProjectRoot '.env'
    if (-not (Test-Path -LiteralPath $taskEnvFile)) {
        Copy-Item -LiteralPath (Join-Path $taskProjectRoot '.env.example') -Destination $taskEnvFile
    }
    Write-Output 'Setup complete. Run .\run.ps1, or .\.venv\Scripts\python.exe scripts\build_index.py.'
} finally {
    Pop-Location
    if ($null -eq $taskPreviousCache) { Remove-Item -LiteralPath Env:UV_CACHE_DIR -ErrorAction SilentlyContinue }
    else { $env:UV_CACHE_DIR = $taskPreviousCache }
}
