<#
.SYNOPSIS
  Stage and commit as devgeektech without Cursor co-author trailers.

.DESCRIPTION
  Without -Branch: checks out master, then commits.
  With -Branch: creates or checks out that branch (from master if new), then commits.
  Uses git write-tree + commit-tree so the message stays clean. Does not push.

.PARAMETER Message
  Commit subject/body (required).

.PARAMETER Branch
  Optional feature branch. Omit to commit on master.

.PARAMETER Paths
  Optional paths to git add. If omitted, stages all changes allowed by .gitignore (git add -A).

.EXAMPLE
  .\scripts\commit.ps1 -Message "Fix HPI stream timeout handling"

.EXAMPLE
  .\scripts\commit.ps1 -Message "WIP stream fix" -Branch feature/stream-fix
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$Message,

    [Parameter(Mandatory = $false)]
    [string]$Branch,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Paths
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$authorName = "devgeektech"
$authorEmail = "development.geektech@gmail.com"
$baseBranch = "master"

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

function Test-LocalBranchExists {
    param([string]$Name)
    & $gitCmd show-ref --verify --quiet "refs/heads/$Name"
    return ($LASTEXITCODE -eq 0)
}

# Select branch before staging
if ($Branch) {
    $Branch = $Branch.Trim()
    if ([string]::IsNullOrWhiteSpace($Branch)) {
        throw "-Branch cannot be empty"
    }
    if (Test-LocalBranchExists $Branch) {
        Write-Host "Checking out existing branch '$Branch' ..."
        Invoke-Git @("checkout", $Branch)
    } else {
        Write-Host "Creating branch '$Branch' from $baseBranch ..."
        Invoke-Git @("checkout", $baseBranch)
        Invoke-Git @("checkout", "-b", $Branch)
    }
} else {
    Write-Host "No -Branch specified; committing on $baseBranch ..."
    Invoke-Git @("checkout", $baseBranch)
}

$currentBranch = (& $gitCmd rev-parse --abbrev-ref HEAD).Trim()

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

tree = run(['write-tree']).stdout.decode().strip()
parent_p = run(['rev-parse', 'HEAD'], check=False)
parent = parent_p.stdout.decode().strip() if parent_p.returncode == 0 else None

args = ['commit-tree', tree]
if parent:
    args += ['-p', parent]
args += ['-F', msg_path]

new = run(args).stdout.decode().strip()
if len(new) != 40:
    sys.stderr.write('commit-tree failed: %r\n' % (new,))
    sys.exit(1)

run(['reset', '--soft', new])
print(new)
"@

try {
    $newSha = & $python -c $py
    if ($LASTEXITCODE -ne 0 -or -not $newSha) {
        throw "Failed to create commit via commit-tree"
    }
    Write-Host "Committed $($newSha.Trim()) on '$currentBranch' as $authorName <$authorEmail>"
    Write-Host ""
    & $gitCmd log -1 --format="author=%an <%ae>%ncommitter=%cn <%ce>%nsubject=%s%n---%n%b"
    Write-Host ""
    & $gitCmd status -sb
    Write-Host ""
    Write-Host "Not pushed. When ready:"
    if ($currentBranch -eq $baseBranch) {
        Write-Host "  git push origin $baseBranch"
    } else {
        Write-Host "  git push -u origin $currentBranch"
        Write-Host "Later merge into master:"
        Write-Host "  .\scripts\merge-to-master.ps1 -Branch $currentBranch"
    }
}
finally {
    Remove-Item -Force $msgFile -ErrorAction SilentlyContinue
}
