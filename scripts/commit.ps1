<#
.SYNOPSIS
  User-friendly commit helper (devgeektech) with optional branch + merge prompts.

.DESCRIPTION
  Interactive (recommended): run with no args and answer the prompts.

    .\scripts\commit.ps1

  Non-interactive (scripts/agent):

    .\scripts\commit.ps1 -Message "Fix"                  # master
    .\scripts\commit.ps1 -Message "Fix" -Branch feat/x   # feature branch
    .\scripts\commit.ps1 -Merge -Branch feat/x           # merge into master

  Never pushes. Repo base branch is master (not main).
#>
param(
    [Parameter(Mandatory = $false)]
    [string]$Message,

    [Parameter(Mandatory = $false)]
    [string]$Branch,

    [Parameter(Mandatory = $false)]
    [switch]$Merge,

    [Parameter(Mandatory = $false)]
    [switch]$NonInteractive,

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
    # Write to host only — do not pollute function return values / $target.
    # Git prints status on stderr; ignore native stderr as terminating errors.
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & $gitCmd @GitArgs 2>&1
        foreach ($line in $output) {
            Write-Host "$line"
        }
        if ($LASTEXITCODE -ne 0) {
            throw "git $($GitArgs -join ' ') failed with exit $LASTEXITCODE"
        }
    }
    finally {
        $ErrorActionPreference = $prev
    }
}

function Test-LocalBranchExists {
    param([string]$Name)
    & $gitCmd show-ref --verify --quiet "refs/heads/$Name" | Out-Null
    return ($LASTEXITCODE -eq 0)
}

function Get-LocalBranches {
    @(
        & $gitCmd for-each-ref --format="%(refname:short)" refs/heads/ 2>$null
    ) | Where-Object { $_ -and $_.Trim() } | ForEach-Object { $_.Trim() }
}

function Read-YesNo {
    param(
        [string]$Prompt,
        [bool]$DefaultYes = $true
    )
    $hint = if ($DefaultYes) { "Y/n" } else { "y/N" }
    while ($true) {
        $raw = Read-Host "$Prompt [$hint]"
        if ([string]::IsNullOrWhiteSpace($raw)) { return $DefaultYes }
        switch -Regex ($raw.Trim().ToLowerInvariant()) {
            '^(y|yes)$' { return $true }
            '^(n|no)$' { return $false }
            default { Write-Host "Please answer yes or no (y/n)." }
        }
    }
}

function Show-PushHints {
    param([string]$OnBranch)
    Write-Host ""
    Write-Host "Done. Not pushed. When ready:"
    if ($OnBranch -eq $baseBranch) {
        Write-Host "  git push origin $baseBranch"
    } else {
        Write-Host "  git push -u origin $OnBranch"
        Write-Host "Or merge later:"
        Write-Host "  .\scripts\commit.ps1 -Merge -Branch $OnBranch"
    }
}

function Invoke-MergeToMaster {
    param(
        [Parameter(Mandatory = $true)][string]$FeatureBranch,
        [string]$MergeMessage
    )
    $FeatureBranch = $FeatureBranch.Trim()
    if ($FeatureBranch -eq $baseBranch) {
        throw "Cannot merge '$baseBranch' into itself."
    }
    if (-not (Test-LocalBranchExists $FeatureBranch)) {
        throw "Branch '$FeatureBranch' does not exist locally."
    }

    $mergeMsg = if ($MergeMessage -and $MergeMessage.Trim()) {
        $MergeMessage.Trim()
    } else {
        "Merge branch '$FeatureBranch' into $baseBranch"
    }

    Write-Host "Checking out $baseBranch ..."
    Invoke-Git @("checkout", $baseBranch)

    Write-Host "Merging '$FeatureBranch' into $baseBranch (--no-ff) ..."
    & $gitCmd -c "user.name=$authorName" -c "user.email=$authorEmail" `
        merge --no-ff $FeatureBranch -m $mergeMsg
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "Merge failed (likely conflicts). Fix files, then:"
        Write-Host "  git add <files>"
        Write-Host "  git -c user.name=$authorName -c user.email=$authorEmail commit --no-edit"
        Write-Host "or abort:  git merge --abort"
        exit $LASTEXITCODE
    }

    Write-Host ""
    Write-Host "Merged '$FeatureBranch' into $baseBranch."
    & $gitCmd log -1 --format="author=%an <%ae>%ncommitter=%cn <%ce>%nsubject=%s"
    Write-Host ""
    & $gitCmd status -sb
    Write-Host ""
    Write-Host "Feature branch '$FeatureBranch' was kept (not deleted)."
    Show-PushHints -OnBranch $baseBranch
}

function Invoke-CommitOnCurrentBranch {
    param(
        [Parameter(Mandatory = $true)][string]$CommitMessage,
        [string[]]$CommitPaths
    )

    if ($CommitPaths -and $CommitPaths.Count -gt 0) {
        $addArgs = @("add", "--") + $CommitPaths
        Invoke-Git $addArgs
    } else {
        Invoke-Git @("add", "-A")
    }

    $status = & $gitCmd status --porcelain
    if (-not $status) {
        Write-Host "Nothing to commit (working tree clean)."
        return $false
    }

    $python = Join-Path $root "venv\Scripts\python.exe"
    if (-not (Test-Path $python)) { $python = "python" }

    $msgFile = Join-Path $env:TEMP ("hpi-commit-msg-{0}.txt" -f [guid]::NewGuid().ToString("N"))
    [System.IO.File]::WriteAllText($msgFile, ($CommitMessage.TrimEnd() + "`n"))

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

def run(args, check=True):
    return subprocess.run([git] + args, capture_output=True, check=check, env=env)

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
        $currentBranch = (& $gitCmd rev-parse --abbrev-ref HEAD).Trim()
        Write-Host ""
        Write-Host "Committed $($newSha.Trim()) on '$currentBranch' as $authorName <$authorEmail>"
        & $gitCmd log -1 --format="author=%an <%ae>%ncommitter=%cn <%ce>%nsubject=%s"
        Write-Host ""
        & $gitCmd status -sb
        return $true
    }
    finally {
        Remove-Item -Force $msgFile -ErrorAction SilentlyContinue
    }
}

function Select-TargetBranchInteractive {
    Write-Host ""
    Write-Host "=== HPI git commit helper ==="
    Write-Host "Author: $authorName <$authorEmail>"
    Write-Host "Base branch: $baseBranch"
    Write-Host ""

    $useMaster = Read-YesNo -Prompt "Commit on branch '$baseBranch'" -DefaultYes $true
    if ($useMaster) {
        Write-Host "Using $baseBranch."
        Invoke-Git @("checkout", $baseBranch)
        return ,$baseBranch
    }

    Write-Host ""
    Write-Host "Existing local branches:"
    $branches = @(Get-LocalBranches)
    for ($i = 0; $i -lt $branches.Count; $i++) {
        $marker = if ($branches[$i] -eq $baseBranch) { " (base)" } else { "" }
        Write-Host ("  [{0}] {1}{2}" -f ($i + 1), $branches[$i], $marker)
    }
    Write-Host ""
    Write-Host "  [N] Create a new branch"
    Write-Host "  [E] Use an existing branch from the list"
    Write-Host ""

    while ($true) {
        $choice = (Read-Host "Create new or use existing? (N/E)").Trim().ToLowerInvariant()
        if ($choice -match '^(n|new)$') {
            while ($true) {
                $name = (Read-Host "New branch name (e.g. feature/my-change)").Trim()
                if (-not $name) {
                    Write-Host "Name cannot be empty."
                    continue
                }
                if ($name -eq $baseBranch) {
                    Write-Host "Use master by answering Yes on the first question instead."
                    continue
                }
                    if (Test-LocalBranchExists $name) {
                    Write-Host "Branch '$name' already exists. Checking it out."
                    Invoke-Git @("checkout", $name)
                    return ,$name
                }
                Write-Host "Creating '$name' from $baseBranch ..."
                Invoke-Git @("checkout", $baseBranch)
                Invoke-Git @("checkout", "-b", $name)
                return ,$name
            }
        }
        if ($choice -match '^(e|existing)$') {
            while ($true) {
                $pick = (Read-Host "Enter branch number or exact name").Trim()
                if ($pick -match '^\d+$') {
                    $idx = [int]$pick - 1
                    if ($idx -ge 0 -and $idx -lt $branches.Count) {
                        $name = $branches[$idx]
                        Invoke-Git @("checkout", $name)
                        return ,$name
                    }
                    Write-Host "Invalid number. Pick 1-$($branches.Count)."
                    continue
                }
                if (Test-LocalBranchExists $pick) {
                    Invoke-Git @("checkout", $pick)
                    return ,$pick
                }
                Write-Host "Unknown branch '$pick'. Try again."
            }
        }
        Write-Host "Please enter N (new) or E (existing)."
    }
}

# ----- Non-interactive merge -----
if ($Merge) {
    if (-not $Branch -or [string]::IsNullOrWhiteSpace($Branch.Trim())) {
        throw "Merge requires -Branch. Example:`n  .\scripts\commit.ps1 -Merge -Branch feature/my-change"
    }
    Invoke-MergeToMaster -FeatureBranch $Branch -MergeMessage $Message
    exit 0
}

# ----- Interactive wizard (default: run with no -Message) -----
if (-not $NonInteractive -and [string]::IsNullOrWhiteSpace($Message)) {
    $target = Select-TargetBranchInteractive

    Write-Host ""
    while ($true) {
        $Message = Read-Host "Commit message"
        if (-not [string]::IsNullOrWhiteSpace($Message)) { break }
        Write-Host "Message cannot be empty."
    }

    $ok = Invoke-CommitOnCurrentBranch -CommitMessage $Message -CommitPaths $Paths
    if (-not $ok) {
        exit 0
    }

    if ($target -ne $baseBranch) {
        Write-Host ""
        $doMerge = Read-YesNo -Prompt "Merge '$target' into $baseBranch now?" -DefaultYes $false
        if ($doMerge) {
            Invoke-MergeToMaster -FeatureBranch $target
            exit 0
        }
        Show-PushHints -OnBranch $target
        exit 0
    }

    Show-PushHints -OnBranch $baseBranch
    exit 0
}

# ----- Non-interactive commit -----
if (-not $Message -or [string]::IsNullOrWhiteSpace($Message.Trim())) {
    throw "Commit requires -Message (or run .\scripts\commit.ps1 with no args for interactive mode)."
}

if ($Branch) {
    $Branch = $Branch.Trim()
    if ([string]::IsNullOrWhiteSpace($Branch)) { throw "-Branch cannot be empty" }
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
$ok = Invoke-CommitOnCurrentBranch -CommitMessage $Message -CommitPaths $Paths
if ($ok) {
    Show-PushHints -OnBranch $currentBranch
}
