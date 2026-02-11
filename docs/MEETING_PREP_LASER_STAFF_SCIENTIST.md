# Meeting prep: staff scientist review of the “step-index MMF + wide linewidth” approach

This is a **one-page-ish** preparation sheet for a technical meeting focused on validating the physics and the
practicality of using:

- a **step-index multimode fiber** (MMF) as an illumination homogenizer, and
- a **few-nm spectral linewidth** laser (2–5 nm class)

to reduce speckle for **short exposures** (e.g. 500 µs) in a multifocus / widefield excitation context.

Companion notebooks:

- `notebooks/14_stepindex_mmf_spectral_linewidth_physics.py` (derivations + intuition + plots)
- `notebooks/13_cni_2nm_stepindex_spectral_diversity_500us.py` (scenario sweeps)

---

## The single chain to defend

### 1) Fiber specs → intermodal optical-path spread

Given fiber length `L`, NA, and core index `n`, a step-index geometric upper bound gives:

```math
\Delta\mathrm{OPL}
= nL\left(\frac{1}{\cos\theta_{\max}}-1\right)
\approx \frac{\mathrm{NA}^2}{2n}\thinspace L,\qquad
\sin\theta_{\max}\approx \mathrm{NA}/n.
```

![](figures/stepindex_angle_definition.svg)

![](figures/stepindex_opl_geometry.svg)

### 2) Intermodal spread → speckle spectral correlation width

```math
\Delta\lambda_c \sim \frac{\lambda^2}{\Delta\mathrm{OPL}}.
```

![](figures/phase_decorrelation_lambda.svg)

![](figures/spectral_correlation_width_cartoon.svg)

![](figures/two_spikes_same_paths_different_speckle.svg)

### 3) Source span → effective number of independent patterns (best-case)

```math
N_\lambda \approx \frac{\Delta\lambda_{\mathrm{src}}}{\Delta\lambda_c},\qquad
C \approx \frac{1}{\sqrt{N_\lambda}}.
```

![](figures/speckle_contrast_vs_bins.svg)

---

## Concrete numbers to sanity-check in the meeting

Using the notebook-13 defaults (λ=640 nm, L=3 m, NA=0.22, n≈1.46):

- ΔOPL ≈ **50 mm** (vacuum-equivalent path)
- Δτ = ΔOPL/c ≈ **170 ps**
- Δλ_c ≈ **0.008 nm** (order of 10⁻² nm)

Then:

- If Δλ_src ≈ 2 nm (instantaneous): N ≈ 250 → C ≈ 0.06
- If Δλ_src ≈ 5 nm: N ≈ 600 → C ≈ 0.04

This scaling summary is often the fastest way to sanity-check “does 0.01 nm really matter?”

![](figures/delta_lambda_c_scaling.svg)

These are “optimistic continuum” values; mode-weighting and partial correlation reduce the benefit.

---

## Whiteboard conversion: Δλ/λ = Δf/f and coherence length

This is the “1/320” line from the whiteboard discussion:

- If λ=640 nm and Δλ=2 nm, then Δλ/λ ≈ 1/320.
- Since f=c/λ, small changes obey Δf/f ≈ Δλ/λ.

The conversion chain:

![](figures/linewidth_to_coherence.svg)

---

## Two “physics pictures” to align on

### Picture A: delay vs coherence envelope

If intermodal delay Δτ is much larger than the source coherence time τ_c, modal interference terms vanish.

![](figures/coherence_vs_delay.svg)

### Picture B: step-index vs graded-index (why SI is good for this)

Step-index fibers have large intermodal delay spread; graded-index fibers intentionally *reduce* it.

![](figures/ray_paths_step_vs_grin.svg)

---

## The gotcha to discuss explicitly: “instantaneous linewidth” vs OSA linewidth

What matters for a 500 µs exposure is the spectrum present *within that 500 µs*.

A laser can show “2–3 nm” on an OSA because:

- it is truly multi-mode simultaneously (good for spectral diversity), or
- it mode-hops across time and the OSA averages it (bad if hopping is slower than exposure).

**Meeting goal:** agree on what measurement (or vendor spec) is credible for “instantaneous linewidth on 500 µs”.

---

## Sharp questions to bring (not fluffy)

### A) About the fiber model

- Is ΔOPL≈(NA²/2n)L a reasonable bound for a 400 µm, NA 0.22 fiber?
- Is this particular “homogenizing fiber” strongly graded-index? If so, what “modal_delay_scale” is realistic?

### B) About the laser

- For a 2–5 nm rated diode module at ~2 W, is it typically multi-longitudinal-mode *simultaneously*?
- What would you expect the longitudinal mode spacing to be?
- What are plausible sources of excess intensity noise (RIN) at 500 µs?

### C) About measurement

- What is the simplest trustworthy experiment to test whether the speckle averages down as predicted?
  - e.g. speckle contrast vs adding a narrow bandpass filter (reduces Δλ_src), or
  - speckle contrast vs fiber length (changes ΔOPL), or
  - swap step-index vs graded-index fiber.

---

## If you get only one actionable outcome

Walk out with **one concrete falsifiable experiment**:

> Measure speckle contrast (same optics) while varying:
> (a) a narrow bandpass filter (reduces Δλ_src), or
> (b) fiber length (changes ΔOPL), or
> (c) swap SI vs GI fiber.

If the observed C follows the predicted scaling trends, the approach is on solid ground.

