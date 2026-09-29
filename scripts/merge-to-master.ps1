<#
.SYNOPSIS
  Merge a feature branch into master (local only; no push).

.PARAMETER Branch
  Feature branch to merge into master (required).

.EXAMPLE
  .\scripts\merge-to-master.ps1 -Branch feature/stream-fix
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$Branch
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$baseBranch = "master"
$Branch = $Branch.Trim()
if ([string]::IsNullOrWhiteSpace($Branch)) {
    throw "-Branch cannot be empty"
}
if ($Branch -eq $baseBranch) {
    throw "Cannot merge '$baseBranch' into itself. Pass a feature branch name."
}

$gitCmd = "C:\Program Files\Git\cmd\git.exe"
if (-not (Test-Path $gitCmd)) { $gitCmd = "git" }

function Invoke-Git {
    param([string[]]$GitArgs)
    & $gitCmd @GitArgs
    if ($LASTEXITCODE -ne 0) {
        throw "git $($GitArgs -join ' ') failed with exit $LASTEXITCODE"
    }
}

& $gitCmd show-ref --verify --quiet "refs/heads/$Branch"
if ($LASTEXITCODE -ne 0) {
    throw "Branch '$Branch' does not exist locally. Create/commit it first with:`n  .\scripts\commit.ps1 -Message '...' -Branch $Branch"
}

Write-Host "Checking out $baseBranch ..."
Invoke-Git @("checkout", $baseBranch)

Write-Host "Merging '$Branch' into $baseBranch (--no-ff) ..."
& $gitCmd merge --no-ff $Branch -m "Merge branch '$Branch' into $baseBranch"
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Merge failed (likely conflicts). Fix conflicted files, then either:"
    Write-Host "  git add <files>"
    Write-Host "  git commit   # finish the merge"
    Write-Host "or abort with:"
    Write-Host "  git merge --abort"
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Merged '$Branch' into $baseBranch."
& $gitCmd log -1 --format="author=%an <%ae>%ncommitter=%cn <%ce>%nsubject=%s"
Write-Host ""
& $gitCmd status -sb
Write-Host ""
Write-Host "Feature branch '$Branch' was kept (not deleted)."
Write-Host "Not pushed. When ready: git push origin $baseBranch"
