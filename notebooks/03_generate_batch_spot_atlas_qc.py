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
# # Batch spot-atlas QC (PowerPoint)
#
# This notebook reproduces the old MATLAB-style **spot atlas** QC, but using the
# outputs of this Python pipeline.
#
# Typical workflow:
#
# 1) Run a batch integrated job (`drivers/run_integrated.py`) with
#    `batch_aggregate_spots: true` in the config.
# 2) Point this notebook at the resulting batch directory.
# 3) Generate:
#    - a representative slide PNG (quick sanity)
#    - an aggregate PPTX covering all detected spots

# %%
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

if "ipykernel" in sys.modules:
    try:
        if matplotlib.get_backend().lower() == "agg":
            matplotlib.use("module://matplotlib_inline.backend_inline", force=True)
    except Exception:
        pass

import matplotlib.pyplot as plt
import pandas as pd

try:
    from IPython.display import display  # type: ignore
except Exception:  # pragma: no cover
    # When run as a plain script, `display` may be unavailable.
    def display(x):  # type: ignore
        print(x)


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

from qc_spot_atlas import SpotAtlasParams, build_spot_atlas_pptx, render_atlas_page_png  # noqa: E402

# %% [markdown]
# ## Choose a batch directory
#
# Set this to a real batch folder under `$BIOIMG_DATA_ROOT/runs/`.

# %%
bioimg_root_raw = os.environ.get("BIOIMG_DATA_ROOT")
if not bioimg_root_raw:
    raise RuntimeError("BIOIMG_DATA_ROOT is not set. See docs/SETUP_WINDOWS.md.")
BIOIMG_DATA_ROOT = Path(bioimg_root_raw).expanduser().resolve()

# Example:
batch_dir = BIOIMG_DATA_ROOT / "runs" / "20260126T172653Z__integrated_batch"

if not batch_dir.exists():
    raise FileNotFoundError(
        f"batch_dir not found: {batch_dir}\n"
        "Update batch_dir to point to an existing batch output folder."
    )

print("batch_dir:", batch_dir)

# %% [markdown]
# ## Load aggregate spots table
#
# Prefer `spots_aggregate.parquet` if it exists.

# %%
agg_path = batch_dir / "spots_aggregate.parquet"
if not agg_path.exists():
    raise FileNotFoundError(
        f"Missing {agg_path}.\n"
        "Re-run the integrated batch with: batch_aggregate_spots: true"
    )

spots = pd.read_parquet(agg_path)
print("spots rows:", len(spots))
spots.head()

# %% [markdown]
# ## Quick summaries

# %%
print("spot channels:", sorted({int(c) for c in spots["spot_channel"].dropna().tolist()}))

# Spots per input
if "input_path" in spots.columns:
    counts = spots.groupby("input_path").size().sort_values(ascending=False)
    display(counts.head(10))

# Intensity distribution
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.hist(spots["intensity"].astype(float).values, bins=60)
ax.set_xlabel("u0 (intensity)")
ax.set_ylabel("count")
ax.set_title("Detected spots intensity histogram")
plt.show()

# %% [markdown]
# ## Render one representative slide image (fast sanity check)
#
# This generates a PNG for the first 15 spots after filtering + sorting.

# %%
params = SpotAtlasParams(
    spots_per_slide=15,
    u0_min=30.0,
    sort_by="intensity",
    fixed_clim=None,  # or (125, 175) to match an old dataset
)

subset = spots[spots["intensity"].astype(float) >= params.u0_min].copy()
subset = subset.sort_values("intensity", ascending=False).head(params.spots_per_slide)

# Minimal state for one page
from qc_spot_atlas import _LRUImageCache  # type: ignore

png_bytes = render_atlas_page_png(
    subset,
    page_title=f"{batch_dir.name} (representative page)",
    params=params,
    image_cache=_LRUImageCache(2),
    manifest_cache={},
    fixed_clim_state={},
)

# Display the PNG
from PIL import Image
import io

im = Image.open(io.BytesIO(png_bytes))
fig, ax = plt.subplots(figsize=(12, 6.5))
ax.imshow(im)
ax.axis("off")
plt.show()

# %% [markdown]
# ## Build the aggregate PPTX (slow, but the main deliverable)
#
# This writes a single PPTX with all detected spots (filtered by `u0_min`).

# %%
out_pptx = batch_dir / "qc_spot_atlas_batch.pptx"

# Optional grouping (examples):
# group_by = "condition"
# group_by = "spot_channel"
# group_by = None

group_by = None

build_spot_atlas_pptx(
    spots,
    out_pptx=out_pptx,
    params=params,
    group_by=group_by,
    deck_title=batch_dir.name,
)

print("wrote:", out_pptx)

