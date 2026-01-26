# Spot detection

This repo’s spot detection (“Slice0”) is deliberately split into **two stages**:

1. **Candidate generation** using a **TrackMate-style Laplacian-of-Gaussian (LoG) detector**.
2. **Per-candidate photometry + acceptance** using the **in5/out0 mask logic** and the
   experimentally calibrated mean intensity threshold (**kept unchanged**).

That separation is intentional: it lets us match **TrackMate/Fiji** for the *geometry* of
candidate detection, while keeping your microscope/dye-specific *intensity physics* logic
as the final arbiter of “real spot vs. not”.

---

## Stage A — Candidate generation (TrackMate-style LoG)

### What we match from TrackMate

Slice0 ports the behavior of TrackMate’s LoG detector (Fiji → TrackMate → *LoG detector*):

- **Kernel construction**: LoG kernel tuned to a **target blob radius** (in calibrated units),
  with TrackMate’s normalization so peak values are comparable across image calibration.
- **FFT convolution**: convolution is performed with FFT and returns an output of the **same
  size** as the input plane.
- **Non-maximum suppression**: candidates are *strict* local maxima in a **3×3 neighborhood**
  (TrackMate uses a rectangle neighborhood of radius 1 pixel).

Optional TrackMate switches are also supported (via config keys):

- median filtering (`spot_do_median_filter`)
- subpixel localization (`spot_do_subpixel_localization`)

### Parameter mapping (TrackMate ↔ this repo)

TrackMate’s LoG detector exposes an **estimated blob diameter** in the GUI, but internally
stores a **radius** (half the diameter). In TrackMate CLI code, the “diameter” argument is
explicitly halved before being stored as `KEY_RADIUS`.

In this repo:

- `spot_radius_nm` (if set) is interpreted as TrackMate’s **radius** in calibrated units.
- If `spot_radius_nm` is not set, we derive a default radius from `(spot_lambda_nm, spot_zR)`
  by treating the optical PSF as a Gaussian beam with waist \(w_0\).

Using the usual Gaussian-beam relation:

\begin{equation}
z_R = \frac{\pi w_0^2}{\lambda}
\quad\Rightarrow\quad
w_0 = \sqrt{\frac{\lambda z_R}{\pi}}.
\end{equation}

We then take

\begin{equation}
\text{radius} \equiv w_0,
\end{equation}

and TrackMate’s displayed diameter becomes

\begin{equation}
d_{\mathrm{TM}} = 2\,w_0.
\end{equation}

TrackMate uses an internal Gaussian scale

\begin{equation}
\sigma = \frac{\text{radius}}{\sqrt{n_{\mathrm{dims}}}},
\end{equation}

so in 2D (\(n_{\mathrm{dims}}=2\)):

\begin{equation}
\sigma_{\mathrm{px}} =
\frac{\text{radius}/\sqrt{2}}{p},
\end{equation}

where \(p\) is the pixel size (same length unit as the radius).

Practical tip: you can extract the XY pixel size from file metadata with:

```
python scripts/inspect_pixel_size.py --input <file.tif|file.ims>
```

### Thresholding (“quality”)

TrackMate’s LoG detector outputs a “quality” value at each detected maximum; it then keeps
spots with

\begin{equation}
q \ge q_{\min}.
\end{equation}

Here:

- `quality` in `spots.parquet` is the **LoG response at the candidate maximum**.
- `spot_q_min` is the **candidate threshold** \(q_{\min}\) (TrackMate’s “threshold”).

### Masks

After local maxima are found, Slice0 can restrict candidates by:

- an optional `valid_mask` (AOI/illumination/etc. mask; nonzero pixels are allowed), and/or
- an optional nuclear mask (`nuclei_labels > 0`) so candidates are **inside nuclei**.

---

## Stage B — Per-candidate photometry (unchanged)

For each remaining candidate \((y,x)\), Slice0 measures intensity in fixed pixel masks
centered on the integer pixel location:

- `background`: median intensity in a thin ring (`out0`)
- `mean_in5`: mean intensity in a small disk (`in5`)
- `mean_in7`: mean intensity in a slightly larger disk (`in7`)

and then computes:

\begin{equation}
u_0 = \langle I \rangle_{\mathrm{in5}} - \mathrm{median}(I)_{\mathrm{out0}},
\qquad
u_1 = \langle I \rangle_{\mathrm{in7}} - \mathrm{median}(I)_{\mathrm{out0}}.
\end{equation}

The acceptance rule is **unchanged**:

In this repo, the contract column `intensity` is currently set to `u0` (i.e. the background-subtracted mean in the in5 disk).

\begin{equation}
u_0 > u_{0,\min}.
\end{equation}

This is the step that encodes your microscope/dye characterization: the “expected mean pixel
brightness” over the in5 scale for an in-focus, relatively immobile emitter over the exposure.

---

## QC: matching TrackMate GUI settings

If you want to reproduce the same candidate stage in Fiji/TrackMate for QC:

1. Ensure Fiji’s image calibration (pixel size) matches your dataset.
2. In TrackMate → LoG detector, set:
   - **Estimated blob diameter** \(d_{\mathrm{TM}} = 2\,\text{radius}\)
   - **Threshold** \(= \texttt{spot\_q\_min}\)

The notebooks `01_step_by_step_integrated_qc.py` and `04_babysit_spot_detection.py`
print the TrackMate-equivalent **blob diameter** implied by the current config.

---

## Attribution and licensing note

The candidate-generation stage is implemented to be *behaviorally consistent* with TrackMate’s
LoG detector. TrackMate is an open-source Fiji plugin distributed under the **GNU GPL v3**.
This repo includes a Python reimplementation of the TrackMate LoG detector’s kernel + maxima logic
for reproducibility.

This section is **not legal advice**: if you plan to redistribute this code, review license
compatibility and attribution requirements with your institution/maintainers.
