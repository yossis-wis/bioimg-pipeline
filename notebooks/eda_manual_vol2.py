# %% [markdown]
# # THE BIOPHYSICIST’S PYTHON EDA MANUAL
# ## Volume 2: The Anatomy of a Matrix (Slicing, Masks, and ROIs)
# 
# **Format:** Jupytext script (`.py` with `# %%` cell markers).  
# **Output:** Optimized for Black & White Laser Printing.
# 
# ### Introduction: The Coordinate System Shock
# The single greatest source of bugs when moving from MATLAB to Python is matrix indexing. You must rewrite your muscle memory with these two absolute rules of the NumPy `ndarray`:
# 
# 1. **Zero-Indexing:** The first element is at index `0`, not `1`.
# 2. **End-Exclusive Slicing:** A slice of `0:3` means "give me indices 0, 1, and 2." It stops *before* 3.
# 
# ### Visualizing the Grid
# In optics, we think in Cartesian `(X, Y)`. But memory matrices are structured in `(Rows, Columns)`. Therefore, an image array is always indexed as `img[Y, X]`.
# 
# | Python Index `img[Y, X]` | Col 0 (X=0) | Col 1 (X=1) | Col 2 (X=2) | Col 3 (X=3) |
# | :--- | :--- | :--- | :--- | :--- |
# | **Row 0 (Y=0)** | `img[0, 0]` | `img[0, 1]` | `img[0, 2]` | `img[0, 3]` |
# | **Row 1 (Y=1)** | `img[1, 0]` | `img[1, 1]` | `img[1, 2]` | `img[1, 3]` |
# | **Row 2 (Y=2)** | `img[2, 0]` | `img[2, 1]` | `img[2, 2]` | `img[2, 3]` |
# 
# *Notice that the Y-axis points DOWN. Y=0 is the top row of the image.*

# %%
# ==========================================
# THE UNIVERSAL PREAMBLE (Physicist's Edition)
# ==========================================
# We proudly use the star import here to eliminate cognitive inertia during EDA!
from numpy import *                 
from matplotlib.pyplot import *     
import pandas as pd
import seaborn as sns # We can safely import this now!

# -- B&W Printer Plot Configuration --
style.use('default')
rcParams['figure.figsize'] = (7, 4.5)
rcParams['figure.facecolor'] = 'white'
rcParams['axes.facecolor'] = 'white'
rcParams['axes.edgecolor'] = 'black'
rcParams['text.color'] = 'black'
rcParams['image.cmap'] = 'gray_r'  # gray_r = inverted gray (white background, black signal)

print("Volume 2 Preamble Loaded. Entering Matrix Mode.\n")

# %% [markdown]
# ### 1. Generating a "Toy" Sub-Pixel Matrix
# MathWorks documentation often uses small $5 \times 5$ matrices so you can see the actual numbers. Let's create a $10 \times 10$ toy matrix representing a single zoomed-in diffraction spot. We will use core NumPy math functions to inspect it.

# %%
# 1. Create a 10x10 background of 50 photons
# (Rows, Columns) -> (Y, X). Notice the double parentheses!
roi = ones((10, 10)) * 50

# 2. Inject a 3x3 "spot" in the center (Rows 4,5,6 and Cols 4,5,6)
# Remember: 4:7 means indices 4, 5, 6 (stops before 7!)
roi[4:7, 4:7] = 200

# 3. Make the exact center pixel the brightest
roi[5, 5] = 450

print("--- CORE MATRIX FUNCTIONS ---")
print(f"1. Matrix Shape: {roi.shape} (Y-rows, X-columns)")
print(f"2. Data Type:    {roi.dtype} (Float64 by default)")
print(f"3. Max Photons:  {max(roi)} (Notice we don't need np.max!)")
print(f"4. Total Photons:{sum(roi)}")

# Let's print the raw numbers to the console (The ultimate ground truth)
print("\n--- RAW ROI MATRIX ---")
print(roi)

# %% [markdown]
# ### 2. Visualizing the Matrix with Pixel Annotations
# To truly map the numbers to the image, we will plot the matrix and overlay the exact numerical value of the photons on top of each pixel. This is the programmatic equivalent of zooming all the way in with your mouse in the MATLAB Figure Viewer.

# %%
figure(figsize=(6, 6))

# Plot the matrix. interpolation='nearest' prevents anti-aliasing blurring so we see crisp pixels.
imshow(roi, vmin=0, vmax=500, interpolation='nearest')
colorbar(label="Photon Counts", fraction=0.046, pad=0.04)

# Overlay the numerical value on every single pixel using a nested loop
# shape[0] is the number of Rows (Y), shape[1] is the number of Cols (X)
for y in range(roi.shape[0]):
    for x in range(roi.shape[1]):
        # If the pixel is bright (black in gray_r), use white text for contrast.
        text_color = 'white' if roi[y, x] > 250 else 'black'
        text(x, y, str(int(roi[y, x])), 
             ha='center', va='center', color=text_color, fontsize=10, fontweight='bold')

title("Sub-Pixel Matrix Inspection", fontweight='bold')
xlabel("X Index (Columns)")
ylabel("Y Index (Rows)")
# Force tick marks at every pixel
xticks(arange(0, 10))
yticks(arange(0, 10))
tight_layout()
show()

# %% [markdown]
# ### 3. Logical Indexing (Creating a Binary Mask)
# In MATLAB, you threshold an image using `mask = img > 100;`. In Python, the syntax is identical!
# This operation returns a matrix of `True` and `False` (Booleans). In Python, `True` mathematically acts as `1` and `False` acts as `0`.

# %%
# Create a binary mask where pixels are strictly greater than 100
binary_mask = roi > 100

print("--- BINARY MASK ---")
# Multiplying by 1 converts True/False into 1/0 for easy printing to the console
print(binary_mask * 1) 

# Visualize the mask
figure(figsize=(4, 4))
imshow(binary_mask, vmin=0, vmax=1)
title("Binary Mask (Pixels > 100)", fontweight='bold')
xlabel("X Index (Columns)")
ylabel("Y Index (Rows)")
tight_layout()
show()

# %% [markdown]
# ### 4. The `find()` Equivalent: `where()`
# In MATLAB, `[row, col] = find(mask)` returns the coordinates of the `True` pixels.
# In Python, the equivalent is `y_coords, x_coords = where(mask)`. 

# %%
# Extract the Y and X coordinates of our bright spot
y_coords, x_coords = where(binary_mask)

print("--- COORDINATES OF BRIGHT PIXELS ---")
for i in range(len(y_coords)):
    # We can use these coordinates to index back into the original matrix!
    print(f"Pixel {i}: Y={y_coords[i]}, X={x_coords[i]}  -> Intensity: {roi[y_coords[i], x_coords[i]]}")

# Calculate the Center of Mass (Mean of coordinates)
mean_y = mean(y_coords)
mean_x = mean(x_coords)
print(f"\nCenter of Mass of Spot: Y = {mean_y}, X = {mean_x}")

# %% [markdown]
# ### 5. Applying the Mask (Data Extraction & Background Zeroing)
# You can pass a Boolean mask directly into a matrix.
# 
# *   **Extraction:** `roi[binary_mask]` flattens the 2D matrix and returns a 1D list of *only* the values where the mask is True.
# *   **Overwriting:** `roi[~binary_mask] = 0` sets everything that is NOT (`~`) in the mask to zero (background subtraction).

# %%
# 1. Extraction: Get only the photon counts inside the mask
spot_photons = roi[binary_mask]

print("--- EXTRACTED DATA ---")
print(f"1D Array of Spot Photons: {spot_photons}")
print(f"Number of Pixels in Spot: {len(spot_photons)}")
print(f"Mean Intensity of Spot:   {mean(spot_photons):.1f}")

# 2. Overwriting: Zero out the background
# We create a copy so we don't destroy our original 'roi' variable
roi_background_subtracted = copy(roi)

# The '~' symbol is the logical NOT operator in Python (equivalent to MATLAB's '~')
roi_background_subtracted[~binary_mask] = 0

print("\n--- BACKGROUND SUBTRACTED MATRIX ---")
print(roi_background_subtracted)

# %% [markdown]
# ---
# ### End of Volume 2.
# **Your Matrix Anatomy Flashcards:**
# * [ ] **`img.shape`** : Tuple of (Rows, Columns) / (Y, X).
# * [ ] **`img[0, 0]`** : Access the absolute first pixel (Top-Left).
# * [ ] **`img[-1, -1]`** : Access the absolute last pixel (Bottom-Right).
# * [ ] **`img[0:5, 0:5]`** : Slice a 5x5 ROI (Indices 0, 1, 2, 3, 4). Stop index is exclusive!
# * [ ] **`mask = img > thresh`** : Generate a True/False Boolean matrix.
# * [ ] **`y_idx, x_idx = where(mask)`** : Python's `find()`. Returns lists of coordinates.
# * [ ] **`data = img[mask]`** : Extracts only the `True` pixels into a flat 1D array.