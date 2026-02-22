# Supporting note: “wide-linewidth + square step-index MMF” for speckle-reduced widefield excitation (500 µs at 640 nm)

This note is intended as an optional **1–2 page attachment** for vendors / applications engineers.

---

## 1) Summary (what we’re trying to build)

We want a practical, robust way to deliver **homogeneous widefield epi illumination** into an inverted microscope:

- Microscope: **Nikon Eclipse Ti2-E**
- Objective: **100× high-NA (oil)**
- Field at sample: **30×30 µm²**
- We need **widefield epi** only (**not** TIRF, **not** HILO)

Performance targets:

- **640-ish excitation (most demanding)**
  - Wavelength class: **635–655 nm**
  - Power density at sample: **10–30 kW/cm²** over 30×30 µm² (≈ 90–270 mW at sample)
  - Timing: **500 µs pulses** with a clean step (fast TTL and/or analog; rise/fall and jitter ≪ 500 µs)
  - Goal: **no visible speckle / interference artifacts at the sample plane** at 500 µs, ideally even without mechanical scrambling

- **Other desired lines** (less time-critical)
  - 405 / 488 / 561 nm
  - Power density at sample: **≥3 kW/cm²** over 30×30 µm²
  - Exposures: **≥5 ms** are fine
  - A commercial **~10 kHz fiber agitator** is acceptable here (and may be sufficient), though wide linewidth is still preferred

Simple system picture (measurement points highlighted):

![](../figures/wide_linewidth_mmf_widefield_system.svg)

We strongly prefer a solution that is **robust for departmental use** (researchers without an optics background).

---

## 2) The core idea we want vendor feedback on

We are testing whether it’s possible to get very low-speckle widefield excitation by combining:

1) a **broad spectral linewidth / multi-longitudinal-mode laser** (**many spectral spikes present simultaneously**), and  
2) a **large-core, square, step-index multimode fiber** (MMF).

**Definition of “broad” (important):** we do **not** mean a swept source or rapidly tuned single-frequency laser. We want a spectrum that is **instantaneously multi-spike**, ideally with many spikes (order 100+) and with spike powers that are roughly comparable (or smoothly varying / Gaussian-like) across the band — not a spectrum dominated by 1–2 spikes.

**Fiber direction:** square, step-index, large core. Our initial target is:

- core size: **400 µm** (200–400 µm acceptable)
- NA: **higher NA preferred** (e.g. **NA ≈ 0.39** if available; NA ~0.22 is also common)
- length: **~3 m** (a few meters)
- coupling: SMA905 or free-space (we can handle coupling)

A conceptual picture of “many spikes → many speckle realizations” at the MMF exit:

![](../figures/speckle_toy_exit_face_interference.svg)

**Mechanical scrambling note:** we may still use a **~10 kHz fiber agitator** even at 640 nm. At 500 µs, a 10 kHz agitator has a 100 µs period, so the theoretical ceiling is ~5 distinct patterns during a 500 µs exposure. That helps (it multiplies whatever spectral averaging we already have), but it is **not sufficient by itself**, which is why the **instantaneous spectral diversity** is the main requirement.

---

## 3) What we want measured (most useful vendor data)

If you have (or can easily take) any of the following, it would be extremely helpful. We care most about measurements at operating powers/currents relevant to our use case.

### A) Spectrum (instantaneous)
- **OSA traces** of the 635–655 nm source at several drive currents / output powers.
- If possible: confirm the spectrum is **simultaneously multi-spike** (not scanned/swept).
- Any note on whether the spectrum is “many similar spikes” vs “1–2 dominant spikes” is valuable.

### B) Modulation / pulse fidelity
- TTL and analog modulation specs and (ideally) **scope traces** for a **500 µs square pulse**:
  - rise/fall time
  - overshoot / ringing
  - timing jitter / repeatability

### C) Speckle / homogeneity (near-field and sample plane)
We care about two planes:

1) **MMF exit face near-field** (camera image of the fiber face)  
2) **Sample plane / image plane** after an objective (widefield illumination uniformity over ~30×30 µm²)

Helpful outputs include:
- representative images (near-field and/or sample plane) at exposure **500 µs**
- any quantitative speckle metric (e.g. speckle contrast), or even qualitative comparison images

### D) “Multiple diodes combined” option
If you recommend spectral broadening via combining multiple nearby diodes (635/640/645/650/655), any guidance (or an integrated offering) is very welcome.

---

## 4) Questions (what would you recommend / what can you provide?)

1) Does the “instantaneous multi-spike + square step-index MMF” approach sound reasonable for **500 µs** widefield uniformity at ~640 nm?
2) What is the **broadest-linewidth** 635–655 nm source you can offer, and can you share **measured spectra**?
3) Do you see any reason this approach would fail in practice (e.g., insufficient decorrelation, residual interference, dominant spikes, etc.)?
4) What are the **modulation specs** (TTL and analog) for clean **500 µs** steps?
5) Have you measured or can you comment on **illumination homogeneity / speckle** at:
   - the **MMF exit face**, and/or
   - a **widefield sample plane** after an objective?
6) Fiber advice: square vs round, **core size**, **NA** (higher NA preferred), and realistic lengths.
7) If multi-diode combination is the best way to widen the spectrum, can you supply it as an integrated, robust output (ideally into one MMF)?

---

## 5) Integration notes (so we don’t talk past each other)

- We need **widefield epi illumination** (uniform illumination over a defined ROI). We are **not** trying to do TIRF or HILO.
- We are open to either:
  - **critical epi illumination** (imaging the fiber face to the sample), or
  - a **Köhler-like relay** (separate control of field stop and pupil fill).
- We are happy to take guidance on which is more practical and robust for coupling into a Ti2-E.

---

## 6) Procurement constraints

- The most important item is the **laser source** (spectrum + fast modulation). We can source fiber/coupling optics separately.
- If you offer an integrated **engine** (especially modular / user-upgradable to add wavelengths later to a common MMF output), that is preferred.
