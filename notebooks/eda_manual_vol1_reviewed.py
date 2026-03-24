# %% [markdown]
# # THE BIOPHYSICIST’S PYTHON EDA MANUAL
# ## Volume 1: The Cockpit & Matrix Mode
#
# **Format:** Jupytext script (`.py` with `# %%` cell markers).  
# **Output:** Optimized for Black & White Laser Printing.
#
# ### Introduction: Philosophy of the Cockpit
# Python is a Bazaar of different tools. To survive without cognitive overload, we split our workflow into two distinct mental modes:
# 1. **Matrix Mode (The MATLAB Brain):** For optics, images, arrays, and PSFs. We use procedural global functions (e.g., `mean(img)`).
# 2. **Table Mode (The R Brain):** For catalogs of spots, tracks, and statistics. We use Object-Oriented methods (e.g., `df.mean()`).
#
# **The Anti-Chaining Rule:** Software engineers love chaining commands together (`df.dropna().groupby().mean()`). As experimentalists, we reject this. We write step-by-step code, assigning explicit variables so we can double-click and inspect the physical reality of the data at every stage in our Variable Explorer.

# %% [markdown]
# ### 1. The Universal Preamble & Printer Setup
# We reject "hidden" startup scripts because they destroy reproducibility. Every EDA scratchpad must start with this visible block.
#
# *Note: We configure Matplotlib specifically for high-contrast B&W printing (white backgrounds, black text, distinct markers, and grayscale colormaps).*

# %%
# ==========================================
# THE UNIVERSAL PREAMBLE (MATLAB‑mode, but reproducible)
# ==========================================
# Philosophy:
# 1) For *interactive EDA*, we intentionally use wildcard imports for NumPy + pyplot,
#    so you can type mean(), max(), plot(), imshow() like MATLAB.
# 2) We STILL keep module aliases (np, plt, pd) so you can:
#      • use tab completion (np.<TAB>, plt.<TAB>)
#      • disambiguate names when needed (np.sum vs py.sum)
#      • copy/paste code into your pipeline later (your repo uses np/plt/pd)
#
# NOTE: This is for notebooks / scratchpads. Avoid wildcard imports in reusable library code.

import builtins as py  # escape hatch for Python builtins (py.sum, py.max, ...)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns  # optional; enable if installed

# --- MATLAB-like convenience (explicit, visible magic) ---
from numpy import *                 # noqa: F401,F403
from matplotlib.pyplot import *     # noqa: F401,F403

from scipy.ndimage import gaussian_filter

# --- B&W Printer Plot Configuration ---
plt.style.use("grayscale")  # good defaults for black & white printing
plt.rcParams.update({
    "figure.figsize": (7.0, 4.5),
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.5,
    "grid.color": "lightgray",
    "text.color": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "lines.linewidth": 2.0,
    "image.cmap": "gray",   # never use rainbow heatmaps for B&W printing
})

# Reproducible demo outputs
np.random.seed(0)

# Backend choice:
#   • For HTML export / printing: inline
#   • For interactive GUI zoom/pan (Spyder): %matplotlib qt
# %matplotlib inline

print("✅ Cockpit loaded (MATLAB‑mode). B&W printing defaults enabled.")


# %% [markdown]
# ### 1.1 Star‑import “Safety Rails” (read once)
#
# This notebook intentionally uses:
#
# - `from numpy import *`
# - `from matplotlib.pyplot import *`
#
# …because you want **MATLAB‑like low‑inertia EDA**: `mean(x)`, `plot(x)`, `imshow(img)`.
#
# The trade‑off is **name ambiguity** (the reason the wider Python community discourages wildcard imports):
#
# - `sum`, `max`, `min`, `any`, `all` will now be **NumPy** versions, not Python built‑ins.
# - `random` will refer to **NumPy’s random module** (not Python’s `random`).
# - If you copy code into your “real pipeline” later, you’ll typically rewrite calls as `np.mean(...)`, `plt.imshow(...)`, etc.
#
# **Escape hatches (memorize these):**
#
# - Use **Python builtins explicitly** via `py` (we imported `builtins as py`):
#   - `py.sum(list_of_numbers)`
#   - `py.max(list_of_strings)`
# - Check where a name came from:
#   - `mean.__module__`
#   - In Jupyter/IPython: `mean?` (also shows signature + docs)
#

# %%
# %% [SAFETY] Quick namespace sanity check (optional)
import inspect

def _origin(obj) -> str:
    return getattr(obj, "__module__", type(obj).__module__)

names = ["mean", "median", "std", "sum", "max", "min", "plot", "imshow", "hist", "random"]
for n in names:
    obj = globals().get(n, None)
    if obj is None:
        print(f"{n:>8} : (not found)")
    else:
        print(f"{n:>8} : {_origin(obj)}")

# Warn about shadowed builtins that often surprise people
shadowed = []
for n in ["sum", "max", "min", "any", "all", "round"]:
    if globals().get(n, None) is not getattr(py, n):
        shadowed.append(n)

if shadowed:
    print("\n⚠️ Shadowed Python builtins:", ", ".join(shadowed))
    print("   Use py.<name>(...) if you truly want the Python built-in.")
else:
    print("\n✅ No builtin names were shadowed (unexpected in MATLAB‑mode).")


# %% [markdown]
# ### 2. First Contact: The Physical Matrix
# In Python, an image is a NumPy `ndarray`. Remember: **Python arrays are 0-indexed, and slicing is end-exclusive.**
# Let's generate a synthetic camera frame with a baseline offset, a few diffraction-limited spots, and Poisson shot noise.

# %%
# 1. Create a 100×100 pixel sensor with a 100‑photon background
camera_frame_ideal = ones((100, 100)) * 100

# 2. Inject fluorescent spots (Y, X coordinates)
camera_frame_ideal[50, 30] = 3000   # Spot A: Center‑Left
camera_frame_ideal[20, 80] = 1200   # Spot B: Top‑Right
camera_frame_ideal[80, 50] = 800    # Spot C: Bottom‑Center

# 3. Apply the Point Spread Function (PSF) blur
# MATLAB: fspecial + convn  ->  Python: gaussian_filter
camera_frame_blurred = gaussian_filter(camera_frame_ideal, sigma=1.5)

# 4. Add physical photon shot noise
# MATLAB: poissrnd  ->  Python: np.random.poisson (here: random.poisson from NumPy)
camera_frame = random.poisson(camera_frame_blurred)

# --- THE 3 CORE MATRIX COMMANDS ---
print("--- MATRIX DIAGNOSTICS ---")
print(f"1) Shape (Y, X): {camera_frame.shape}")
print(f"2) Max pixel (saturation check): {max(camera_frame.ravel())}")
print(f"3) Mean background (top‑left 10×10): {mean(camera_frame[0:10, 0:10]):.1f}")


# %% [markdown]
# ### 3. Visualizing the 2D Field of View
# When printing, standard color maps (like `viridis`) wash out into muddy gray blobs. By explicitly using `cmap='gray'` and setting strict contrast limits (`vmin` and `vmax`), we ensure the spots punch through the background on paper.
#
# We also overlay annotations using hollow markers with contrasting edges.

# %%
figure(figsize=(6, 6))

# Plot the 2D matrix. vmax clamps the upper contrast limit.
imshow(camera_frame, vmin=80, vmax=500,cmap='gray_r')
colorbar(label="Photon Counts", fraction=0.046, pad=0.04)

# Annotate Spot A (Note: plot takes X, Y. The matrix was Y, X)
# For B&W printing: facecolor='none' makes a hollow circle.
plot(30, 50, marker='o', markersize=25, fillstyle='none', 
     markeredgecolor='white', markeredgewidth=3, label='Target Spot')
plot(30, 50, marker='o', markersize=25,  fillstyle='none', 
     markeredgecolor='black', markeredgewidth=1, linestyle='--')

title("Raw Acquisition (B&W Print Optimized)", fontweight='bold')
xlabel("X Pixels")
ylabel("Y Pixels")
legend(loc='upper right')
tight_layout()
show()

# %% [markdown]
# ### 4. Extracting a 1D Line Profile (The Scalpel)
# A visual learner must see the sub-pixel physics. Inspecting a 2D image is not enough to verify a Point Spread Function. We must slice a 1D profile across our brightest spot (Y=50) to verify its Gaussian shape against the noise floor.
#
# *Notice the MATLAB-style slicing: `camera_frame[50, :]` grabs row 50 across all columns.*

# %%
# Slice a horizontal row through the center of our brightest spot
line_profile = camera_frame[50, :]
x_axis = arange(len(line_profile))

figure(figsize=(8, 4))

# For B&W: Solid black line with explicit circle markers for each pixel
plot(x_axis, line_profile, color='black', marker='o', 
     markerfacecolor='white', markersize=5, label='Raw Pixel Data')

# Add a dashed baseline for the background noise floor
axhline(y=100, color='gray', linestyle='--', linewidth=2, label='Theoretical Background')

title("1D Axial Profile Across Spot A (Y = 50)", fontweight='bold')
xlabel("X Pixel Coordinate")
ylabel("Intensity (Photons)")
xlim(15, 45)  # Zoom in on the spot
grid(True, linestyle=':')
legend()
tight_layout()
show()

# %% [markdown]
# ### 5. The Intensity Histogram (Noise vs. Signal)
# To establish a signal-to-noise threshold, we need the distribution of all pixels. We use `.ravel()` to flatten the 2D matrix into a 1D array for the histogram.
#
# For B&W printing, we use a light gray face color with sharp black edges for the bars.

# %%
figure(figsize=(8, 4))

# Flatten the image and plot the histogram
hist(camera_frame.ravel(), bins=50, facecolor='lightgray', edgecolor='black', linewidth=1.2)

# Draw a vertical line showing a hypothetical detection threshold
axvline(x=150, color='black', linestyle='-.', linewidth=2, label='Detection Threshold (150)')

yscale('log') # Log scale because background pixels vastly outnumber spot pixels
title("Pixel Intensity Distribution", fontweight='bold')
xlabel("Intensity (Photons)")
ylabel("Log(Pixel Count)")
grid(True, linestyle=':', alpha=0.7)
legend()
tight_layout()
show()

# %% [markdown]
# ---
# ### End of Volume 1.
# **Your Muscle Memory Checklist (The "Flashcards"):**
# * [ ] **`img.shape`** : Get the dimensions of your optical frame.
# * [ ] **`max(img)`** : Check for camera saturation.
# * [ ] **`mean(img[y1:y2, x1:x2])`** : Probe a local background region.
# * [ ] **`img[y, :]`** : Slice a 1D line profile.
# * [ ] **`img.ravel()`** : Flatten a 2D image for a histogram.
# * [ ] **`imshow(img, vmin=..., vmax=...)`** : Clamp visual contrast.
