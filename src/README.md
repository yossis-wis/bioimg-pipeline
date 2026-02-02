# src/

This folder contains the core “kernel” implementations used by the drivers and notebooks.

Key modules:

- `slice0_kernel.py` — **Slice0 spot detection**.
  - Stage A: TrackMate-style LoG candidate generation (FFT convolution + 3×3 local maxima + `q_min`).
  - Stage B: fixed-mask photometry (`in5/out0` → `u0_min`) kept unchanged.
- `slice1_nuclei_kernel.py` — **Slice1 nuclei segmentation** (StarDist wrapper).
- `qc_spot_interactive.py` — matplotlib-based interactive QC for Slice0 (sliders for `q_min`, `u0_min`).
- `vis_utils.py` — helper functions for QC overlays/montages.

Illumination / Fourier-optics analysis:

- `excitation_speckle_sim.py` — Fourier-optics toy model for widefield excitation speckle (square ROI + pupil low-pass).
- `illumination_design_params.py` — first-order design calculations (power budget, stop sizing, pupil fill, beam diameters).
- `speckle_diversity_models.py` — bookkeeping for speckle averaging (time, spectral, polarization, angle diversity).

See `docs/SPOT_DETECTION.md` for the spot-detection algorithm details and TrackMate parameter mapping.

