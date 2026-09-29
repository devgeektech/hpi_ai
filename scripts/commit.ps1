<#
.SYNOPSIS
  Stage and commit as devgeektech without Cursor co-author trailers.

.DESCRIPTION
  Uses git write-tree + commit-tree (via Python + Git for Windows) so the
  commit message stays clean. Does not push.

.PARAMETER Message
  Commit subject/body (required).

.PARAMETER Paths
  Optional paths to git add. If omitted, stages all tracked/untracked
  changes allowed by .gitignore (git add -A).

.EXAMPLE
  .\scripts\commit.ps1 -Message "Fix HPI stream timeout handling"
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$Message,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Paths
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$authorName = "devgeektech"
$authorEmail = "development.geektech@gmail.com"

$gitCmd = "C:\Program Files\Git\cmd\git.exe"
$gitBin = "C:\Program Files\Git\bin\git.exe"
if (-not (Test-Path $gitBin)) { $gitBin = $gitCmd }
if (-not (Test-Path $gitCmd)) { $gitCmd = "git" }

function Invoke-Git {
    param([string[]]$GitArgs)
    & $gitCmd @GitArgs
    if ($LASTEXITCODE -ne 0) {
        throw "git $($GitArgs -join ' ') failed with exit $LASTEXITCODE"
    }
}

# Stage
if ($Paths -and $Paths.Count -gt 0) {
    $addArgs = @("add", "--") + $Paths
    Invoke-Git $addArgs
} else {
    Invoke-Git @("add", "-A")
}

$status = & $gitCmd status --porcelain
if (-not $status) {
    Write-Host "Nothing to commit (working tree clean)."
    exit 0
}

$python = Join-Path $root "venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

$msgFile = Join-Path $env:TEMP ("hpi-commit-msg-{0}.txt" -f [guid]::NewGuid().ToString("N"))
# Ensure trailing newline for commit-tree
$normalized = $Message.TrimEnd() + "`n"
[System.IO.File]::WriteAllText($msgFile, $normalized)

$py = @"
import os, subprocess, sys

git = r'''$gitBin'''
msg_path = r'''$msgFile'''
author_name = r'''$authorName'''
author_email = r'''$authorEmail'''

env = os.environ.copy()
env['GIT_AUTHOR_NAME'] = author_name
env['GIT_AUTHOR_EMAIL'] = author_email
env['GIT_COMMITTER_NAME'] = author_name
env['GIT_COMMITTER_EMAIL'] = author_email
# Do not reuse stale dates from the parent shell
env.pop('GIT_AUTHOR_DATE', None)
env.pop('GIT_COMMITTER_DATE', None)

def run(args, check=True, input=None):
    return subprocess.run(
        [git] + args,
        input=input,
        capture_output=True,
        check=check,
        env=env,
    )

# Index must already be staged by the PowerShell caller
tree = run(['write-tree']).stdout.decode().strip()
parent_p = run(['rev-parse', 'HEAD'], check=False)
parent = parent_p.stdout.decode().strip() if parent_p.returncode == 0 else None

with open(msg_path, 'rb') as f:
    msg = f.read()

args = ['commit-tree', tree]
if parent:
    args += ['-p', parent]
args += ['-F', msg_path]

new = run(args).stdout.decode().strip()
if len(new) != 40:
    sys.stderr.write('commit-tree failed: %r\n' % (new,))
    sys.exit(1)

# soft: move HEAD only — keeps working tree; index already matches this tree
run(['reset', '--soft', new])
print(new)
"@

try {
    $newSha = & $python -c $py
    if ($LASTEXITCODE -ne 0 -or -not $newSha) {
        throw "Failed to create commit via commit-tree"
    }
    Write-Host "Committed $($newSha.Trim()) as $authorName <$authorEmail>"
    Write-Host ""
    & $gitCmd log -1 --format="author=%an <%ae>%ncommitter=%cn <%ce>%nsubject=%s%n---%n%b"
    Write-Host ""
    & $gitCmd status -sb
    Write-Host ""
    Write-Host "Not pushed. When ready: git push origin master"
}
finally {
    Remove-Item -Force $msgFile -ErrorAction SilentlyContinue
}
