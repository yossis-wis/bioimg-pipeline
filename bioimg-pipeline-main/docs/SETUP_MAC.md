# bioimg-pipeline — macOS Setup Guide

This document describes how to set up **bioimg-pipeline** on a **macOS workstation** (MacBook Pro / Mac Pro) in a reproducible way. 

**Tested on:** MacBook Pro 16-inch 2019 (Intel Core i7), macOS Sequoia 15.6.1

---

## 0) Core idea: Code vs Data vs Docs (do not mix)

- **Code (git repo):** a local folder, e.g.   
  `~/Code/bioimg-pipeline`

- **Data bench (NOT in git):** a folder in your home directory or external drive, e.g.  
  `~/bioimg-data`  
  (This holds raw staging data, run outputs, caches, models, etc.)

- **Docs (optional):** iCloud/OneDrive/Dropbox folder.   
  **Do not put the git repo inside a synced cloud folder.**

**Rule of thumb:** the repo should always be "zip-able" and small; the bench can be huge. 

---

## 1) Prerequisites

### Required

1. **Git**
   - GitHub Desktop (which you have) includes Git.
   - Alternatively, install via Xcode command line tools:  `xcode-select --install`

2. **Miniforge (recommended) or Miniconda**
   - Miniforge is well-maintained and works great on Intel Macs.
   - Install via Homebrew: `brew install miniforge`
   - Verify:  `conda --version`

3. **Visual Studio Code**
   - Installed ✅

### VS Code extensions (install from VS Code → Extensions)
- **Python** (Microsoft)
- **Pylance** (Microsoft)
- **Jupyter** (Microsoft)

### Remove Standalone Spyder ⚠️

If you have a standalone Spyder installation, **uninstall it** before proceeding: 

1. Open **Finder**
2. Go to **Applications**
3. Find **Spyder** and drag it to Trash
4. Empty Trash
5. Clean up Spyder settings: 
   ```zsh
   rm -rf ~/.spyder-py3
   rm -rf ~/Library/Application\ Support/spyder-py3
   ```

**Why?** The standalone Spyder uses its own bundled Python and won't see your `bioimg-slice0` packages.  Your `environment.yml` already includes Spyder—use that version instead.

---

## 2) Clone the repo

Recommended location for code: `~/Code/`

Open **Terminal** and run:

```zsh
mkdir -p ~/Code
cd ~/Code
git clone https://github.com/yossis-wis/bioimg-pipeline.git
cd bioimg-pipeline
```

*Alternatively, use GitHub Desktop to clone into `~/Code/bioimg-pipeline`.*

---

## 3) Create the data bench folders (outside git)

Choose a data root (example uses `~/bioimg-data`). For large datasets, consider an external SSD. 

Create these folders:

```zsh
mkdir -p ~/bioimg-data/raw_staging
mkdir -p ~/bioimg-data/runs
mkdir -p ~/bioimg-data/cache
mkdir -p ~/bioimg-data/models
```

Notes: 

- `raw_staging/` is where you drop a small test `.tif` during development.
- `runs/` is where each run writes outputs into its own subfolder. 
- `cache/` and `models/` are reserved for later slices.

---

## 4) Set the BIOIMG_DATA_ROOT environment variable

This makes the code portable across machines without changing paths in code.

1. Open your zsh config:
   ```zsh
   nano ~/.zshrc
   ```

2. Add the following line to the bottom of the file:
   ```zsh
   export BIOIMG_DATA_ROOT="$HOME/bioimg-data"
   ```

3. Save and exit (`Ctrl+O`, `Enter`, `Ctrl+X`).

4. **Apply the changes** (or restart terminal):
   ```zsh
   source ~/.zshrc
   ```

### Sanity check

```zsh
echo $BIOIMG_DATA_ROOT
```

Should print:  `/Users/yourusername/bioimg-data`

---

## 5) Create and verify the conda environment

From the repo root (`~/Code/bioimg-pipeline`):

```zsh
cd ~/Code/bioimg-pipeline
conda env create -f environment.yml
conda activate bioimg-slice0
python scripts/verify_setup.py
```

Expected: the script ends with `SETUP OK ✅`.

### Updating later (after `git pull`)

If `environment.yml` changes in the future:

```zsh
conda activate bioimg-slice0
conda env update -f environment.yml --prune
python scripts/verify_setup.py
```

---

## 6) VS Code: select interpreter + run code

1. Open VS Code
2. **File → Open Folder…** → `~/Code/bioimg-pipeline`
3. Select the interpreter:
   - Press `Cmd+Shift+P`
   - Search: **Python: Select Interpreter**
   - Choose: **bioimg-slice0**

VS Code should now show `bioimg-slice0` in the bottom status bar.

---

## 7) Jupyter in VS Code (optional for Slice 0)

If you open a `.ipynb` notebook:

1. Click **Select Kernel** (top-right in the notebook toolbar)
2. Choose the kernel/interpreter for **bioimg-slice0**

Quick sanity cell:

```python
import sys, os
print(sys.executable)
print(os.environ.get("BIOIMG_DATA_ROOT"))
```

You should see the `.../miniforge3/envs/bioimg-slice0/bin/python` path and your data root.

---

## 8) Spyder (from conda environment)

Launch Spyder from within the activated environment:

```zsh
conda activate bioimg-slice0
spyder &
```

The `&` runs Spyder in the background so you can keep using the terminal.

Recommendation:

- Use Spyder as an interactive dev tool if it helps.
- Keep the "authoritative" execution path as command-line drivers (later slices).

---

## 9) Quick reference commands

| Action | Command |
|--------|---------|
| Activate environment | `conda activate bioimg-slice0` |
| Deactivate environment | `conda deactivate` |
| Update environment | `conda env update -f environment.yml --prune` |
| List environments | `conda env list` |
| Pull latest code | `git pull origin main` |
| Check Git status | `git status` |
| Launch Spyder | `spyder &` (while env is active) |
| Launch JupyterLab | `jupyter lab` (while env is active) |

---

## Troubleshooting

### "conda:  command not found"

```zsh
# Re-initialize conda
~/miniforge3/bin/conda init zsh
# Restart terminal
```

### Environment creation fails

```zsh
# Update conda first
conda update conda

# Force recreate
conda env create -f environment.yml --force
```

### VS Code doesn't see the environment

1. Restart VS Code completely
2. Press `Cmd+Shift+P` → "Python: Clear Cache and Reload Window"
3. Try selecting interpreter again

### verify_setup.py not found

The script may not exist yet. Run this sanity check instead:

```zsh
python -c "import numpy, pandas, skimage, tifffile; print('SETUP OK ✅')"
```
