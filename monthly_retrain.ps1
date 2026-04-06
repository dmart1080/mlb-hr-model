# =============================================================================
# MLB HR Model — Monthly Retrain Script
# =============================================================================
# Pulls fresh prediction CSVs from VPS, rebuilds current season months,
# retrains the model, and deploys to the VPS in one command.
#
# Usage:
#   .\monthly_retrain.ps1                        # default: refresh current season
#   .\monthly_retrain.ps1 -FullRebuild           # wipe + rebuild all 5 seasons
#   .\monthly_retrain.ps1 -SkipVpsPull           # skip SSH steps (local only)
#   .\monthly_retrain.ps1 -DryRun                # show what would happen, no changes
#
# Requirements:
#   - SSH key auth configured for VPS (no password prompt)
#   - VPS_HOST set below
#   - Python venv at .venv\Scripts\python.exe
# =============================================================================

param(
    [switch]$FullRebuild,
    [switch]$SkipVpsPull,
    [switch]$DryRun
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
$VPS_HOST     = "root@5.161.226.89"       
$VPS_DIR      = "/root/mlb-hr-model"
$PYTHON       = ".venv\Scripts\python.exe"
$CURRENT_YEAR = (Get-Date).Year
$LOG_FILE     = "logs\monthly_retrain_$(Get-Date -Format 'yyyy-MM-dd').log"

# ── HELPERS ───────────────────────────────────────────────────────────────────
function Log {
    param([string]$msg, [string]$color = "White")
    $ts = Get-Date -Format "HH:mm:ss"
    Write-Host "[$ts] $msg" -ForegroundColor $color
    Add-Content -Path $LOG_FILE -Value "[$ts] $msg"
}

function Run {
    param([string]$cmd, [string]$desc)
    Log "→ $desc" "Cyan"
    if ($DryRun) {
        Log "  [DRY RUN] Would run: $cmd" "Yellow"
        return
    }
    $output = Invoke-Expression $cmd 2>&1
    $output | ForEach-Object { Add-Content -Path $LOG_FILE -Value $_ }
    if ($LASTEXITCODE -ne 0) {
        Log "✗ FAILED: $desc" "Red"
        Log "  Exit code: $LASTEXITCODE" "Red"
        exit 1
    }
    Log "✓ Done: $desc" "Green"
}

function RunSsh {
    param([string]$remote_cmd, [string]$desc)
    Run "ssh $VPS_HOST `"$remote_cmd`"" $desc
}

# ── SETUP ─────────────────────────────────────────────────────────────────────
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
New-Item -ItemType Directory -Force -Path "data\predictions" | Out-Null

Log "=" * 60 "Magenta"
Log "MLB HR MODEL — MONTHLY RETRAIN" "Magenta"
Log "Date      : $(Get-Date -Format 'yyyy-MM-dd HH:mm')" "Magenta"
Log "Mode      : $(if ($FullRebuild) { 'FULL REBUILD' } elseif ($DryRun) { 'DRY RUN' } else { 'CURRENT SEASON REFRESH' })" "Magenta"
Log "VPS       : $VPS_HOST" "Magenta"
Log "Log file  : $LOG_FILE" "Magenta"
Log "=" * 60 "Magenta"

# ── STEP 1: Pull prediction CSVs from VPS ─────────────────────────────────────
if (-not $SkipVpsPull) {
    Log ""
    Log "STEP 1/5 — Pulling prediction CSVs from VPS" "Yellow"
    Log "These are used by isotonic recalibration in train.py"

    if ($DryRun) {
        Log "  [DRY RUN] Would run: scp $VPS_HOST`:$VPS_DIR/data/predictions/*.csv data\predictions\" "Yellow"
    } else {
        $result = scp "${VPS_HOST}:${VPS_DIR}/data/predictions/*.csv" "data\predictions\" 2>&1
        if ($LASTEXITCODE -eq 0) {
            $count = (Get-ChildItem "data\predictions\*.csv" -ErrorAction SilentlyContinue).Count
            Log "✓ Pulled prediction CSVs ($count files in data\predictions\)" "Green"
        } else {
            Log "⚠ SCP failed — continuing without fresh prediction CSVs" "Yellow"
            Log "  Isotonic recalibration will use existing local CSVs" "Yellow"
        }
    }
} else {
    Log "STEP 1/5 — Skipping VPS pull (--SkipVpsPull)" "Yellow"
}

# ── STEP 2: Delete stale parquets ────────────────────────────────────────────
Log ""
Log "STEP 2/5 — Clearing stale feature parquets" "Yellow"

if ($FullRebuild) {
    Log "Full rebuild: deleting ALL monthly parquets and combined table"
    if (-not $DryRun) {
        Get-ChildItem "data\processed\train_table_*-*-*_to_*-*-*.parquet" -ErrorAction SilentlyContinue | Remove-Item
        Remove-Item "data\processed\train_table_2021_2026_full.parquet" -ErrorAction SilentlyContinue
        Remove-Item "data\processed\train_table_TEST*.parquet" -ErrorAction SilentlyContinue
        Log "✓ Deleted all monthly parquets" "Green"
    } else {
        Log "  [DRY RUN] Would delete all data\processed\train_table_*_to_*.parquet" "Yellow"
    }
} else {
    Log "Partial refresh: deleting only $CURRENT_YEAR monthly parquets"
    if (-not $DryRun) {
        Get-ChildItem "data\processed\train_table_$CURRENT_YEAR*.parquet" -ErrorAction SilentlyContinue | Remove-Item
        Remove-Item "data\processed\train_table_2021_2026_full.parquet" -ErrorAction SilentlyContinue
        $deleted = (Get-ChildItem "data\processed\train_table_$CURRENT_YEAR*.parquet" -ErrorAction SilentlyContinue).Count
        Log "✓ Deleted $CURRENT_YEAR parquets (combined table will be rebuilt)" "Green"
    } else {
        Log "  [DRY RUN] Would delete data\processed\train_table_$CURRENT_YEAR*.parquet" "Yellow"
    }
}

# ── STEP 3: Rebuild feature tables ───────────────────────────────────────────
Log ""
Log "STEP 3/5 — Rebuilding feature tables" "Yellow"
if ($FullRebuild) {
    Log "This will take 2-4 hours for a full 5-season rebuild"
} else {
    Log "This will take 20-40 min for current season months only"
}

Run "$PYTHON -m src.features.build_features_multi_season" "Build feature tables"

# ── STEP 4: Retrain model ────────────────────────────────────────────────────
Log ""
Log "STEP 4/5 — Retraining model" "Yellow"
Run "$PYTHON -m src.model.train" "Train model"

# ── STEP 5: Commit and push ───────────────────────────────────────────────────
Log ""
Log "STEP 5/5 — Committing and deploying to VPS" "Yellow"

$commit_msg = if ($FullRebuild) {
    "chore: full rebuild retrain $(Get-Date -Format 'yyyy-MM-dd')"
} else {
    "chore: monthly retrain with fresh $CURRENT_YEAR data $(Get-Date -Format 'yyyy-MM-dd')"
}

if (-not $DryRun) {
    # Stage model and any changed source files
    Run "git add models\hr_model_lightgbm_calibrated_2021_2026.joblib" "Stage model"

    # Also stage any modified source files
    $changed = git diff --name-only 2>$null
    if ($changed) {
        Run "git add src\" "Stage source changes"
    }

    # Check if there's anything to commit
    $status = git status --porcelain 2>$null
    if ($status) {
        Run "git commit -m `"$commit_msg`"" "Commit"
        Run "git push" "Push to GitHub"

        if (-not $SkipVpsPull) {
            Log ""
            Log "Deploying to VPS..."
            RunSsh "cd $VPS_DIR && source .venv/bin/activate && git pull" "Pull on VPS"
            Log "✓ VPS updated — new model active on next cron run" "Green"
        }
    } else {
        Log "Nothing to commit — model unchanged" "Yellow"
    }
} else {
    Log "  [DRY RUN] Would commit: $commit_msg" "Yellow"
    Log "  [DRY RUN] Would push to GitHub and pull on VPS" "Yellow"
}

# ── SUMMARY ───────────────────────────────────────────────────────────────────
Log ""
Log "=" * 60 "Magenta"
Log "RETRAIN COMPLETE" "Green"
Log "Log saved to: $LOG_FILE" "Magenta"
Log "=" * 60 "Magenta"

# Print last model metrics from log
Log ""
Log "Model metrics from this run:" "Cyan"
if (-not $DryRun) {
    $train_log = Get-Content $LOG_FILE -ErrorAction SilentlyContinue |
        Where-Object { $_ -match "ROC-AUC|Log loss|Top 10%|Top 1%" } |
        Select-Object -Last 6
    $train_log | ForEach-Object { Log "  $_" "White" }
}
