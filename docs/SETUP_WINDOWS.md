# bioimg-pipeline — Windows Setup Guide

This document describes how to set up **bioimg-pipeline** on a **new Windows workstation** in a way that is reproducible and avoids OneDrive/file-lock issues.

---

## 0) Core idea: Code vs Data vs Docs (do not mix)

- **Code (git repo):** a local folder, e.g.  
  `C:\Code\bioimg-pipeline`

- **Data bench (NOT in git):** a large local drive or network location, e.g.  
  `D:\bioimg-data`  
  (This holds raw staging data, run outputs, caches, models, etc.)

- **Docs (optional):** OneDrive folder (Word/PDF/PowerPoint), e.g.  
  `OneDrive\BioimgPipeline_Docs`  
  **Do not put the git repo inside OneDrive.**

**Rule of thumb:** the repo should always be “zip-able” and small; the bench can be huge.

---

## 1) Install prerequisites

### Required
1. **Git for Windows**
   - You can use GitHub Desktop if you prefer, but Git itself is the foundation.

2. **Miniconda**
   - Install Miniconda (recommended over full Anaconda).
   - **Do NOT add conda to PATH** during installation (avoids conflicts).

3. **Visual Studio Code**
   - Install VS Code.

### VS Code extensions (install from VS Code → Extensions)
- **Python** (Microsoft)
- **Pylance** (Microsoft)
- **Jupyter** (Microsoft)

> Optional: GitHub Desktop (nice UI for commit/push, not required).

---

## 2) Clone the repo

Recommended location for code: `C:\Code\`

Open **Command Prompt** (or Git Bash) and run:

```bat
cd C:\Code
git clone https://github.com/yossis-wis/bioimg-pipeline.git
cd bioimg-pipeline
```

---

## 3) Create the data bench folders (outside git)

Choose a data root (example uses `D:\bioimg-data`).

Create these folders:

- `D:\bioimg-data\raw_staging`
- `D:\bioimg-data\runs`
- `D:\bioimg-data\cache`
- `D:\bioimg-data\models`

Notes:

- `raw_staging\` is where you drop a small test `.tif` during development.
- `runs\` is where each run writes outputs into its own subfolder.
- `cache\` and `models\` are reserved for later slices.

---

## 4) Set the BIOIMG_DATA_ROOT environment variable (User)

This makes the code portable across machines without changing paths in code.

Open **PowerShell** and run:

```powershell
[Environment]::SetEnvironmentVariable("BIOIMG_DATA_ROOT","D:\bioimg-data","User")
```

Then **close and reopen** terminals/VS Code so the new variable is visible.

Sanity checks:

- In PowerShell:

  ```powershell
  echo $env:BIOIMG_DATA_ROOT
  ```

- In Command Prompt:

  ```bat
  echo %BIOIMG_DATA_ROOT%
  ```

---

## 5) Create and verify the conda environment

> Run conda commands in **Anaconda Prompt (Miniconda3)** or a terminal where conda is available.

From repo root (`C:\Code\bioimg-pipeline`):

```bat
cd C:\Code\bioimg-pipeline
conda env create -f environment.yml
conda activate bioimg-slice0
python scripts/verify_setup.py
```

Expected: the script ends with `SETUP OK ✅`.

### Updating later (after `git pull`)

If `environment.yml` changes in the future:

```bat
conda activate bioimg-slice0
conda env update -f environment.yml --prune
python scripts/verify_setup.py
```

---

## 6) VS Code: select interpreter + run code

1. Open VS Code
2. **File → Open Folder…** → `C:\Code\bioimg-pipeline`
3. Select the interpreter:
   - Press `Ctrl+Shift+P`
   - Search: **Python: Select Interpreter**
   - Choose: **bioimg-slice0**

### Terminal behavior (important)

VS Code may open a terminal that does **not** auto-activate conda.

If you open a terminal and you do not see `(bioimg-slice0)` in the prompt, run:

```bat
conda activate bioimg-slice0
```

That’s OK—manual activation is normal on Windows.

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

You should see the `...miniconda3\envs\bioimg-slice0\python.exe` path and your data root.

---

## 8) Spyder (optional)

If Spyder is included in your environment, you can launch it from an activated terminal:

```bat
conda activate bioimg-slice0
spyder
```

Recommendation:

- Use Spyder as an interactive dev tool if it helps.
- Keep the “authoritative” execution path as command-line drivers (later slices).

---

## 9) About `environment.yml` location (root vs configs)

Keeping `environment.yml` at the **repo root** is standard practice because:

- It is part of “how to install the project,” not a runtime parameter.
- It makes setup commands simple: `conda env create -f environment.yml`.

The `configs/` folder is reserved for **runtime configuration** (e.g., `configs/dev.yaml`).

If you ever move the environment file, you must update this doc and your commands accordingly.

---

## 10) GitHub Desktop warning about LF ↔ CRLF line endings

If GitHub Desktop shows something like:

> “This diff contains a change in line endings from LF to CRLF”

This is common on Windows and not a functional problem.

Recommended practice:

- Prefer **LF** for repo text files (Markdown/YAML) to reduce cross-platform friction.
- In VS Code you can switch line endings via the bottom-right status bar (“CRLF/LF”).

If you already committed CRLF, it’s usually fine—just aim for consistency going forward.

---

## 11) What should NOT be in this repo

Do **not** put any of these inside the git repo:

- Raw microscope data
- Run output folders
- Cache/model artifacts
- OneDrive-synced directories

Those belong in `BIOIMG_DATA_ROOT` (your bench), not in git.
