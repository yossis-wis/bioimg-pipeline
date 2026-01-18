# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Review a run folder
#
# Load manifests, spot tables, and QC images from a completed run.

# %%
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

def _find_repo_root() -> Path:
    cwd = Path.cwd().resolve()
    for parent in (cwd, *cwd.parents):
        if (parent / "src").is_dir():
            return parent
    raise RuntimeError(
        "Could not find repo root (expected a src/ directory). "
        f"Start this notebook from the repo root or a subdirectory. cwd={cwd}"
    )


REPO_ROOT = _find_repo_root()
sys.path.insert(0, str(REPO_ROOT / "src"))

# %% [markdown]
# ## Inputs

# %%
bioimg_data_root_raw = os.environ.get("BIOIMG_DATA_ROOT")
if not bioimg_data_root_raw:
    raise RuntimeError("BIOIMG_DATA_ROOT is not set. See docs/SETUP_WINDOWS.md.")
bioimg_data_root = Path(bioimg_data_root_raw)
run_dir = bioimg_data_root / "runs" / "<timestamp>__integrated"

spots_path = run_dir / "spots.parquet"
manifest_path = run_dir / "run_manifest.yaml"
qc_glob = "qc_overlay*.png"

if not run_dir.exists():
    raise FileNotFoundError(f"Run directory not found: {run_dir}")

# %% [markdown]
# ## Load run manifest

# %%
if manifest_path.exists():
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    print("Run manifest keys:", sorted(manifest.keys()))
else:
    print("No run_manifest.yaml found")

# %% [markdown]
# ## Load spots table

# %%
if not spots_path.exists():
    raise FileNotFoundError(f"Missing spots table: {spots_path}")
spots_df = pd.read_parquet(spots_path)
spots_df.head()

# %% [markdown]
# ## Summary metrics

# %%
summary = {
    "num_spots": int(len(spots_df)),
}
if "nucleus_label" in spots_df.columns:
    summary["num_nuclei"] = int(spots_df["nucleus_label"].nunique())
if "snr" in spots_df.columns:
    summary["median_snr"] = float(spots_df["snr"].median())
summary

# %% [markdown]
# ## Display QC overlays

# %%
qc_paths = sorted(run_dir.glob(qc_glob))
if not qc_paths:
    print("No QC overlay images found")
else:
    for qc_path in qc_paths:
        img = plt.imread(qc_path)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img)
        ax.set_title(qc_path.name)
        ax.axis("off")
        plt.show()
