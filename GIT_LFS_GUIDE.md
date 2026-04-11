# Git LFS Guide for Large Binary Assets

This project uses machine-learning models, video footage, and NumPy arrays that can easily exceed GitHub's 100 MB per-file limit. This guide explains how to handle those assets safely.

---

## Why pushes fail with HTTP 408

A push that contains hundreds of megabytes often times out before GitHub finishes receiving the pack (you'll see `error: RPC failed; HTTP 408` in the output). The root cause is almost always that large binary files were accidentally committed.

---

## What is already ignored

The root `.gitignore` excludes the following heavy assets so they are never committed accidentally:

| Pattern | What it covers |
|---|---|
| `slowfast-model/` | SlowFast model weights and checkpoints |
| `sample-footage/` | Raw video samples used for testing |
| `data/inputs/*` | Input video / image files |
| `data/outputs/*` | Pipeline output files |
| `*.npy` | NumPy array files |
| `*.mp4` | Video files |
| `*.dav` | DVR/camera recording files |
| `demo_output_*` | Demo output directories/files |
| `*.pt` / `*.pth` | PyTorch model weight files |
| `*.onnx` / `*.pkl` | Other model serialization formats |

---

## If you need to version-control large assets: use Git LFS

[Git Large File Storage (LFS)](https://git-lfs.com/) replaces large files in the repository with small pointer files and stores the actual content on a separate server. This keeps the Git history fast while still allowing large assets to be versioned.

### One-time setup (per machine)

```bash
# Install Git LFS
git lfs install

# Tell LFS which file types to track (run once per new pattern)
git lfs track "*.npy"
git lfs track "*.mp4"
git lfs track "*.dav"
git lfs track "slowfast-model/**"
git lfs track "sample-footage/**"

# Make sure .gitattributes is committed
git add .gitattributes
git commit -m "Configure Git LFS tracking"
```

### After that, add files normally

```bash
git add slowfast-model/
git commit -m "Add SlowFast model weights (via LFS)"
git push
```

Git will upload the binary content to LFS storage and only store the pointer in the regular Git history.

---

## Recommended Git settings for large pushes

Even with LFS, pushing a large pack over a slow or unstable connection can still time out. Apply these settings once per machine to raise the timeouts:

```bash
git config --global http.postBuffer 524288000   # 500 MiB send buffer
git config --global http.lowSpeedLimit 0        # disable low-speed abort
git config --global http.lowSpeedTime 999999    # effectively infinite timeout
```

---

## Removing accidentally committed large files

If a large file has already been committed (and the push fails or is rejected), remove it from the last commit:

```bash
# Unstage the directory/file from Git tracking, keep it on disk
git rm -r --cached slowfast-model/
git rm --cached "*.npy" "*.mp4" "*.dav"

# Commit the removal
git add .gitignore
git commit -m "Remove large assets from tracking"

# Now push should succeed
git push -u origin <your-branch>
```

To rewrite deeper history (multiple commits), use `git filter-repo` or BFG Repo Cleaner instead of `git rm`.

---

## Further reading

- [GitHub: About large files](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
- [Git LFS official site](https://git-lfs.com/)
- [BFG Repo Cleaner](https://rtyley.github.io/bfg-repo-cleaner/)
