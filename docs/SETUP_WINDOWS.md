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

```plaintext
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
conda activate bioimg-pipeline
python scripts/verify_setup.py
```

Expected: the script ends with `SETUP OK ✅`.

### Updating later (after `git pull`)

If `environment.yml` changes in the future:

```bat
conda activate bioimg-pipeline
conda env update -f environment.yml --prune
python scripts/verify_setup.py
```

If `verify_setup.py` reports a `typing_extensions` problem (or VS Code Jupyter fails to start),
see **Troubleshooting** below.

---

## 6) VS Code: select interpreter + run code

1. Open VS Code
2. **File → Open Folder…** → `C:\Code\bioimg-pipeline`
3. Select the interpreter:
   - Press `Ctrl+Shift+P`
   - Search: **Python: Select Interpreter**
   - Choose: **bioimg-pipeline**

### Set the default terminal in VS Code to Command Prompt (recommended)

1. Open **VS Code**
2. Go to **File → Preferences → Settings**
3. Search for `default profile windows`
4. Find **"Terminal › Integrated: Default Profile Windows"**
5. Select **"Command Prompt"** from the dropdown

After this change, new terminals in VS Code will open as Command Prompt with the conda environment auto-activated. 

---

## 7) Jupyter in VS Code (optional for Slice 0)

If you open a `.ipynb` notebook:

1. Click **Select Kernel** (top-right in the notebook toolbar)
2. Choose the kernel/interpreter for **bioimg-pipeline**

Quick sanity cell:

```python
import sys, os
print(sys.executable)
print(os.environ.get("BIOIMG_DATA_ROOT"))
```

You should see the `...miniconda3\envs\bioimg-pipeline\python.exe` path and your data root.

---

## 8) Spyder (optional)

Spyder is included in `environment.yml`. Launch it from an activated terminal:

```bat
conda activate bioimg-pipeline
spyder
```

Recommendation:

- Use Spyder as an interactive dev tool if it helps.
- Keep the “authoritative” execution path as command-line drivers (later slices).

---

## Troubleshooting

### VS Code Jupyter kernel fails with: "TypeAliasType could not be imported from typing_extensions"

**Symptom**

VS Code shows a kernel error similar to:

- `TypeAliasType could not be imported from typing_extensions`

**Cause**

The pip install step for TensorFlow 2.13 can **downgrade** `typing_extensions` to 4.5.0
(TensorFlow pins `<4.6` in its metadata). Recent **Jupyter / Spyder / Pylance** expects a newer
`typing_extensions` that provides `TypeAliasType`.

**Fix (recommended)**

```bat
conda activate bioimg-pipeline
python scripts/fix_typing_extensions.py
python scripts/verify_setup.py
```

**Alternative (direct pip)**

```bat
conda activate bioimg-pipeline
python -m pip install -U "typing-extensions>=4.7"
```

You may see a pip warning like:

- `tensorflow-intel 2.13.0 requires typing-extensions<4.6.0`

In practice, this repo has been validated with newer `typing_extensions`; treat the warning as informational.

After the fix, reload VS Code:

- `Ctrl+Shift+P` → **Developer: Reload Window**

### verify_setup.py fails with: "No module named 'numba'" (StarDist)

**Symptom**

- `python scripts/verify_setup.py` prints an error like:
  - `ERROR: import stardist failed: No module named 'numba'`
  - (or `ERROR: import numba failed: No module named 'numba'`)

**Cause**

`stardist` relies on `numba`. On some machines (or if the environment was created before `environment.yml` included `numba`), that dependency may be missing.

**Fix**

```bat
conda activate bioimg-pipeline
conda env update -f environment.yml --prune
python scripts/verify_setup.py
```

If it still fails, install `numba` directly from conda-forge:

```bat
conda activate bioimg-pipeline
conda install -c conda-forge numba
python scripts/verify_setup.py
```
