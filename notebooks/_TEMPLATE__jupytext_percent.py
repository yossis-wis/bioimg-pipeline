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
# # Notebook template
#
# Copy this file, update the title, and start adding cells below.

# %%
# Imports
from __future__ import annotations

from pathlib import Path

import matplotlib

if "ipykernel" in sys.modules:
    try:
        if matplotlib.get_backend().lower() == "agg":
            matplotlib.use("module://matplotlib_inline.backend_inline", force=True)
    except Exception:
        pass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %% [markdown]
# ## Configuration
#
# Define paths and any constants you need here.

# %%
# Example placeholders
repo_root = Path(".").resolve()

# %% [markdown]
# ## Analysis
#
# Add your analysis cells here.

# %%
# Example cell
x = np.linspace(0, 1, 100)
y = np.sin(2 * np.pi * x)

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(x, y)
ax.set_title("Template plot")
ax.set_xlabel("x")
ax.set_ylabel("sin(2πx)")
plt.show()

