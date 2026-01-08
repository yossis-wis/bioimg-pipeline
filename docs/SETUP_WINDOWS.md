# bioimg-pipeline

Code for a reproducible bioimaging analysis pipeline.

## Repository vs Data (important)

- **Code (this repo):** `C:\Code\bioimg-pipeline`
- **Data bench (NOT in git):** e.g. `D:\bioimg-data`
- **Docs (optional):** OneDrive folder (do not place the git repo inside OneDrive)

## New workstation setup (Windows)

### 1) Install prerequisites
- Git for Windows (GitHub Desktop optional)
- Miniconda (do not add to PATH)
- VS Code + extensions:
  - Python (Microsoft)
  - Jupyter (Microsoft)

### 2) Clone the repo
git clone <repo-url>
cd bioimg-pipeline

### 3) Create the data bench folders (outside git)

Choose a data root (example uses D:\bioimg-data).

Create these folders:

- D:\bioimg-data\raw_staging
- D:\bioimg-data\runs
- D:\bioimg-data\cache
- D:\bioimg-data\models

Notes:
- raw_staging/ is where you drop a test.tif during development.
- runs/ is where outputs go (each run creates a subfolder).
- cache/ and models/ are reserved for later slices.

### 4) Set the BIOIMG_DATA_ROOT environment variable (User)

Open PowerShell and run:

[Environment]::SetEnvironmentVariable("BIOIMG_DATA_ROOT","D:\bioimg-data","User")


Important:
- After setting this, close and reopen terminals / VS Code so the new variable is visible.

Quick check (PowerShell):

echo $env:BIOIMG_DATA_ROOT

### 5) Create the conda environment

Recommended: run conda commands from “Anaconda Prompt / Miniconda Prompt”.

From the repo root:

cd C:\Code\bioimg-pipeline
conda env create -f environment.yml
conda activate bioimg-slice0


If the environment already exists and environment.yml changed:

conda env update -f environment.yml --prune
conda activate bioimg-slice0

### 6) Verify the setup (required)

Still in the activated env:

python scripts\verify_setup.py


Expected success indicators:
- conda_env: bioimg-slice0
- BIOIMG_DATA_ROOT: D:\bioimg-data (or your chosen path)
- imports OK
- write test OK
- ends with: SETUP OK ✅

### 7) VS Code setup (recommended)

1. Open VS Code
2. File → Open Folder… → C:\Code\bioimg-pipeline
3. Select interpreter:
  - Ctrl+Shift+P → Python: Select Interpreter
  - choose the interpreter that includes (bioimg-slice0)

Recommended setting (auto-activate env in new terminals):
  - Settings → search: python.terminal.activateEnvironment
  - set to true

Terminal note:
- If conda is “not recognized” in PowerShell, that’s okay.
  - Use Anaconda/Miniconda Prompt for conda commands, or
  - Use Command Prompt inside VS Code terminals.

### 8) (Optional) Jupyter smoke test in VS Code

You can confirm the notebook kernel uses the env:
1. Ctrl+Shift+P → Jupyter: Create New Blank Notebook
2. Click Select Kernel (top right) → choose bioimg-slice0
3. Run:

import sys, os
print(sys.executable)
print(os.environ.get("BIOIMG_DATA_ROOT"))

You should see the ...miniconda3\envs\bioimg-slice0\python.exe path and your bench root.

### 9) What NOT to do

- ❌ Do not put the repo inside OneDrive/Dropbox.
- ❌ Do not commit raw data or run outputs into git.
- ❌ Do not hardcode absolute machine paths inside code (use BIOIMG_DATA_ROOT + relative paths).