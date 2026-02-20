# Supporting note: “wide-linewidth + square step-index MMF” for speckle-free widefield excitation (500 µs at 640 nm)

This note is intended as an optional **1–2 page attachment** for vendors / applications engineers.

---

## 1) Summary (what we’re trying to build)

We want a practical, robust way to deliver **homogeneous widefield epi illumination** into an inverted microscope:

- Microscope: **Nikon Eclipse Ti2-E**
- Objective: **100× high-NA (oil)**
- Field at sample: **30×30 µm²**

Performance targets:

- **640-ish excitation (most demanding)**
  - Wavelength range of interest: **635–655 nm**
  - Power density at sample: **10–30 kW/cm²** over 30×30 µm² (≈ 90–270 mW at sample)
  - Timing: **500 µs pulses** with a clean step (fast TTL and/or analog; rise/fall and jitter ≪ 500 µs)
  - Goal: **no visible speckle / interference artifacts at the sample plane** at 500 µs, ideally **without mechanical scrambling**

- **Other desired lines** (full system, less time-critical)
  - 405 / 488 / 561 nm
  - Power density at sample: **≥3 kW/cm²** over 30×30 µm²
  - Exposures: **≥5 ms** are fine
  - For these, adding a commercial **~10 kHz fiber agitator** is acceptable (but wide linewidth is still preferred)

We strongly prefer a solution that is **robust for departmental use** (researchers without an optics background).

---

## 2) The core idea we want vendor feedback on

We are testing whether it’s possible to get speckle-free (or very low-speckle) widefield excitation by combining:

1) a **broad spectral linewidth / multi-longitudinal-mode laser** (many spectral spikes present simultaneously), and
2) a **large-core, square, step-index multimode fiber** (MMF), e.g. **200–400 µm** core and **a few meters** long.

The desired laser characteristic is **not** a swept source or a rapidly tuned single-frequency laser. We specifically want a spectrum that is **instantaneously multi-spike**, ideally with spikes of comparable (or smoothly varying / Gaussian-like) power across the band.

A conceptual picture:

![](../figures/speckle_toy_exit_face_interference.svg)

Intuition:

- Each spectral spike produces its **own** speckle realization after the MMF.
- If the spikes are sufficiently decorrelated by the fiber’s intermodal optical-path spread, the observed intensity tends toward an **incoherent sum** of many realizations.
- This can reduce speckle contrast even for short exposures where mechanical averaging is not available.

---

## 3) Prior art / evidence we can share

We have compiled a small excerpt (figures + a representative diode spec sheet) in **`papers.zip`** (not tracked in git because this repo is text-only):

- `papers.zip/papers/figures.pdf` and `papers.zip/papers/figures.docx`
- Laser spec examples:
  - `papers.zip/papers/Mockl2023laser638nm.pdf` (Lasertack LAB-638-1000)
  - `papers.zip/papers/Ries2020laser638nm.pdf` / `papers.zip/papers/KwakwFrench2016laser638nm.pdf` (HL63193MG diode)

Key points from the literature excerpt:

- **Ries et al., Biomed Opt Express (2020)**, “Cost-efficient open source laser engine for microscopy.”
  - Shows speckle behavior for **square-core MMF** and multiple wavelengths.
  - Notably, the **638 nm diode** case appears substantially more homogeneous than shorter wavelengths, and speckle contrast depends on diode operating point.

- **Almahayni et al., J Phys Chem A (2023)**, “Simple, Economic, and Robust Rail-Based Setup for Super-Resolution Localization Microscopy.”
  - Uses a **square-core MMF** and compares **speckle contrast** with and without fiber shaking.
  - The red line (638 nm) shows lower speckle contrast than 561 nm even without shaking.

- **Kwakwa et al., J Biophotonics (2016)**, “easySTORM: a robust, lower-cost approach to localisation and TIRF microscopy.”
  - Supplementary figure compares **static optical fiber** vs **vibrating despeckler**, across different MMF core sizes.
  - Larger-core fibers (e.g. 200–400 µm) look closer to uniform even before adding vibration.

Representative “wide-linewidth diode module” spec (example only, from the excerpt):

- **Lasertack LAB-638-1000** (example 1 W-class 638 nm module)
  - Linewidth: **≤ 5 nm (2 nm typical)**
  - Modulation options: **analog, TTL, PWM**
  - Modulation bandwidth: **DC–500 kHz**
  - Rise/fall time: **~1.4 µs**

We are not assuming these exact parts are the correct answer—rather, this is meant to illustrate the type of spectrum + modulation behavior that seems promising.

---

## 4) What we want from a vendor / applications engineer

We would greatly appreciate any of the following:

1) **Recommendation:** Does the “wide-linewidth + square step-index MMF” approach sound reasonable for **500 µs** illumination uniformity at ~640 nm?
2) **Spectrum data:** any OSA traces showing the **instantaneous** spectrum (multi-spike). If you have data vs current/power, even better.
3) **Homogeneity data:** any measurements of:
   - fiber **near-field** uniformity,
   - sample-plane or image-plane uniformity after an objective,
   - or any speckle contrast characterization.
4) **Modulation performance:** TTL and analog step behavior (especially for a **500 µs** square pulse). Any scope traces are useful.
5) **Fiber guidance:** recommended fiber geometry (square vs round), NA, core size, and realistic lengths.
6) **System architecture:** if broadening the spectrum via **multiple nearby diodes** (635/640/645/650/655) is more robust than “one very broad diode,” we’d like your recommendation.

---

## 5) Integration notes (so we don’t talk past each other)

- We need **widefield epi illumination** (uniform illumination over a defined ROI). We are **not** trying to do TIRF or HILO.
- We are open to either:
  - **critical epi illumination** (imaging the fiber face to the sample), or
  - a **Köhler-like relay** (separate control of field stop and pupil fill).

We are happy to take guidance on which is more practical and robust for coupling into a Ti2-E.

---

## 6) Procurement constraints

- The most important item is the **laser source** (spectrum + fast modulation). We can source fiber/coupling optics separately.
- If possible, we are aiming for **< US$1k** for a 640-ish source, but we understand that integrated multi-line systems may cost more.
- A modular / upgradeable system (add wavelengths later, common MMF output) would be ideal.
