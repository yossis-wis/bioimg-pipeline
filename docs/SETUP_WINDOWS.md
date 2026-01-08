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
```bash
git clone <repo-url>
cd bioimg-pipeline

### 3) Create the data bench folders
Create (example):

D:\bioimg-data\raw_staging
D:\bioimg-data\runs
D:\bioimg-data\cache
D:\bioimg-data\models

### 4) Set environment variable

PowerShell:
[Environment]::SetEnvironmentVariable("BIOIMG_DATA_ROOT","D:\bioimg-data","User")

### 5) Create and verify conda environment
conda env create -f environment.yml
conda activate bioimg-slice0
python scripts/verify_setup.py

### 6) VS Code

Open folder: C:\Code\bioimg-pipeline
Select interpreter: bioimg-slice0