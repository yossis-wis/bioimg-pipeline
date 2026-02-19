# Spot detection

This repo's spot detection ("Slice0") is deliberately split into **two stages**:

1. **Candidate generation** using a **TrackMate-style Laplacian-of-Gaussian (LoG) detector**.
2. **Per-candidate photometry + acceptance** using the **in5/out0 mask logic** and the
   experimentally calibrated mean intensity threshold (**kept unchanged**).

That separation is intentional: it lets us match **TrackMate/Fiji** for the *geometry* of
candidate detection, while keeping your microscope/dye-specific *intensity physics* logic
as the final arbiter of "real spot vs. not".

---

## Stage A — Candidate generation (TrackMate-style LoG)

### What we match from TrackMate

Slice0 ports the behavior of TrackMate's LoG detector (Fiji → TrackMate → *LoG detector*):

- **Kernel construction**: LoG kernel tuned to a **target blob radius** (in calibrated units),
  with TrackMate's normalization so peak values are comparable across image calibration.
- **FFT convolution**: convolution is performed with FFT and returns an output of the **same
  size** as the input plane.
- **Non-maximum suppression**: candidates are *strict* local maxima in a **3×3 neighborhood**
  (TrackMate uses a rectangle neighborhood of radius 1 pixel).

Optional TrackMate switches are also supported (via config keys):

- median filtering (`spot_do_median_filter`)
- subpixel localization (`spot_do_subpixel_localization`)

### Parameter mapping (TrackMate ↔ this repo)

TrackMate's LoG detector exposes an **estimated blob diameter** in the GUI, but internally
stores a **radius** (half the diameter). In TrackMate CLI code, the "diameter" argument is
explicitly halved before being stored as `KEY_RADIUS`.

In this repo:

- `spot_radius_nm` (if set) is interpreted as TrackMate's **radius** in calibrated units (nm in our configs).
- If `spot_radius_nm` is not set, we derive a default radius from `(spot_lambda_nm, spot_zR)` using a
  **legacy optics-inspired parameterization**:

```math
r \equiv \sqrt{\frac{\lambda z_R}{\pi}}\thinspace.
```

We then use

```math
d_{\mathrm{TM}} = 2r
```

as the TrackMate GUI **estimated blob diameter**.

TrackMate uses an internal Gaussian scale

```math
\sigma = \frac{r}{\sqrt{n_{\mathrm{dims}}}},
\qquad
\sigma_{\mathrm{px}} = \frac{\sigma}{p},
```

so in 2D ($`n_{\mathrm{dims}} = 2`$):

```math
\sigma_{\mathrm{px}} = \frac{r}{\sqrt{2}\thinspace p},
```

where $p$ is the pixel size (same length unit as $r$).

**Optics note (optional).** If you model the in-focus PSF intensity as a Gaussian beam

```math
I(r) = I_0\thinspace\exp\!\left(-\frac{2r^2}{w_0^2}\right),
```

then the equivalent Gaussian standard deviation is $`\sigma = w_0/2`$. Matching this to TrackMate's
$\sigma = r/\sqrt{2}$ (2D) implies $`r = w_0/\sqrt{2}`$ and therefore

```math
d_{\mathrm{TM}} = \sqrt{2}\thinspace w_0
```

in 2D. (This repo does **not** enforce a particular optics convention; the quantity that matters for
TrackMate matching is the **radius $r$ actually used by the detector**.)

Practical tip: you can extract the XY pixel size from file metadata with:

```bash
python scripts/inspect_pixel_size.py --input <file.tif|file.ims>
```

You can also set:

- `spot_pixel_size_nm: auto`

in your YAML config to make the drivers infer pixel size from metadata (and warn if a numeric
`spot_pixel_size_nm` disagrees strongly with metadata).

### Thresholding ("quality")

TrackMate's LoG detector outputs a "quality" value at each detected maximum; it then keeps
spots with

```math
q \ge q_{\min}.
```

Here:

- `quality` in `spots.parquet` is the **LoG response at the candidate maximum**.
- `spot_q_min` is the candidate threshold $`q_{\min}`$ (TrackMate's "threshold").

### Masks

After local maxima are found, Slice0 can restrict candidates by:

- an optional `valid_mask` (AOI/illumination/etc. mask; nonzero pixels are allowed), and/or
- an optional nuclear mask (`nuclei_labels > 0`) so candidates are **inside nuclei**.

---

## Stage B — Per-candidate photometry (unchanged)

For each remaining candidate $(y,x)$, Slice0 measures intensity in fixed pixel masks
centered on the integer pixel location:

- `background`: median intensity in a thin ring (`out0`)
- `mean_in5`: mean intensity in a small disk (`in5`)
- `mean_in7`: mean intensity in a slightly larger disk (`in7`)

and then computes:

```math
u_0 = \langle I \rangle_{\mathrm{in5}} - \mathrm{median}(I)_{\mathrm{out0}},
\qquad
u_1 = \langle I \rangle_{\mathrm{in7}} - \mathrm{median}(I)_{\mathrm{out0}}.
```

The acceptance rule is **unchanged**:

In this repo, the contract column `intensity` is currently set to `u0`
(i.e. the background-subtracted mean in the in5 disk).

```math
u_0 > u_{0,\min}.
```

This is the step that encodes your microscope/dye characterization: the "expected mean pixel
brightness" over the in5 scale for an in-focus, relatively immobile emitter over the exposure.

---

## QC: matching TrackMate GUI settings

If you want to reproduce the same candidate stage in Fiji/TrackMate for QC:

1. Ensure Fiji's image calibration (pixel size) matches your dataset.
2. In TrackMate → LoG detector, set:
   - **Estimated blob diameter** $`d_{\mathrm{TM}} = 2r`$
   - **Threshold** $`= \mathtt{spot\_{q\_{min}}}`$

The notebooks `01_step_by_step_integrated_qc.py` and `04_babysit_spot_detection.py`
print the TrackMate-equivalent **blob diameter** implied by the current config.

---

## Multi-channel spot detection (DNA + protein)

The integrated driver supports multi-channel runs by allowing:

- `channel_spots` to be a **list** (e.g. `[1, 2]`).
- Per-channel overrides via `spot_params_by_channel`.

For each spot channel `ch`, the driver constructs a `Slice0Params` instance by:

1. Reading the top-level `spot_*` keys as defaults.
2. Applying `spot_params_by_channel[ch]` (if provided).

This is useful when your DNA-locus spots and protein spots have very different
intensity distributions (so you want different $`u_{0,\min}`$), or when you
want a “LoG-only” first pass for the DNA locus:

- Set `spot_u0_min: 0.0` for that channel.
- Keep `spot_q_min` set to your TrackMate LoG threshold.

Example snippet (TrackMate diameter $`d_{\mathrm{TM}} = 0.54\,\mu\mathrm{m}`$,
threshold 6.182):

```yaml
channel_spots: [1, 2]

spot_params_by_channel:
  1:  # DNA locus (mNeonGreen)
    spot_radius_nm: 270.0   # radius = d/2 = 0.27 µm = 270 nm
    spot_q_min: 6.182       # TrackMate “threshold” (quality)
    spot_u0_min: 0.0        # LoG-only (no calibrated photometry threshold)
  2:  # protein (e.g. Msn2-Halo)
    spot_u0_min: 30.0       # keep your calibrated u0 threshold
```

Note: it is valid for `channel_nuclei` to also appear in `channel_spots`
(e.g. if your nuclei marker and DNA locus signal share the same acquisition
channel).

---

## Attribution and licensing note

The candidate-generation stage is implemented to be *behaviorally consistent* with TrackMate's
LoG detector. TrackMate is an open-source Fiji plugin distributed under the **GNU GPL v3**.
This repo includes a Python reimplementation of the TrackMate LoG detector's kernel + maxima logic
for reproducibility.

This section is **not legal advice**: if you plan to redistribute this code, review license
compatibility and attribution requirements with your institution/maintainers.
