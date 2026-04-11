# Contributing Guide

## Repository Rules for Large Files

Large binary files (model weights, videos, datasets) **must not** be committed to this repository. The `.gitignore` is configured to block common large-file types automatically.

### What is ignored
| Pattern | Reason |
|---|---|
| `slowfast-model/` | SlowFast model directory |
| `sample-footage/` | Raw video footage |
| `*.mp4`, `*.avi`, `*.dav`, `*.mov` | Video files |
| `*.pt`, `*.pth`, `*.onnx`, `*.pkl` | ML model weights |
| `*.npy` | NumPy arrays |
| `demo_output_*` | Demo output artifacts |
| `data/inputs/*`, `data/outputs/*` | Runtime data (keep `.gitkeep`) |
| `__pycache__/`, `*.pyc` | Python bytecode |

### Before you push: check for large files
```powershell
# Windows PowerShell
git ls-files | % { Get-Item $_ -EA SilentlyContinue } |
  Sort-Object Length -Descending |
  Select-Object -First 20 FullName, @{n="MB";e={[math]::Round($_.Length/1MB,2)}}
```
```bash
# Linux / macOS / Git Bash
git ls-files | xargs -I{} du -sh {} 2>/dev/null | sort -rh | head -20
```

If any file is **>50 MB**, do **not** push it. Remove it with:
```bash
git rm --cached <large-file>
echo "<large-file>" >> .gitignore
git add .gitignore
git commit --amend --no-edit   # if it was in your last commit
```

## Git configuration for this repository

Run these once to avoid push timeouts on larger (but allowed) commits:

```bash
git config --global http.postBuffer 524288000   # 500 MiB (mebibytes) send buffer
git config --global http.lowSpeedLimit 0        # disable low-speed abort
git config --global http.lowSpeedTime  999999   # effectively no timeout
```

## Pushing to a feature branch

1. Verify the remote branch exists or create it:
   ```bash
   git fetch origin
   git push -u origin feature/<your-branch>
   ```

2. Confirm the push succeeded by checking the remote tip:
   ```bash
   git fetch origin
   git log --oneline -3 origin/feature/<your-branch>
   # Your latest commit SHA should appear at the top
   ```

3. If the push returns HTTP 408 (timeout) with a pack > ~50 MB, first
   identify and remove large files (see above), then retry.

## ML pipeline assets

Place large assets in these **local-only** directories (already in `.gitignore`):

| Asset type | Directory |
|---|---|
| Videos / footage | `ml_pipeline/src/data/videos/` |
| Images | `ml_pipeline/src/data/images/` |
| Trained models | `ml_pipeline/src/models/` |
| Inference outputs | `ml_pipeline/src/outputs/` |
| SlowFast model | `slowfast-model/` |
| Raw footage | `sample-footage/` |

Empty-directory placeholders (`.gitkeep` files) are committed so the
folder structure is preserved in a fresh clone without any binary assets.
