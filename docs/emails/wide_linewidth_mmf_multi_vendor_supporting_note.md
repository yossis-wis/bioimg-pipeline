# Supporting note: wide-linewidth + square step-index MMF for speckle-reduced widefield epi excitation (≤500 µs at ~640 nm)

This is an optional **1–2 page attachment** for vendors / applications engineers.

---

## 0) Main question (what we want you to answer)

**Main question:** what is the **multimode** laser source with the **broadest instantaneous spectral linewidth (FWHM)** you can offer in the **635–655 nm** class (ideally centered ~646–650 nm, spanning as much of 635–655 nm as practical), and have you verified that **this broad source + a square step-index multimode fiber (MMF)** can give a **homogeneous field at the MMF exit face** even **without** mechanical agitation?

**What we mean by “broad” (important):** an **instantaneously multi-spike / multi-longitudinal-mode** spectrum (order 100+ simultaneously present spikes), with spike powers roughly comparable or smoothly varying (Gaussian-like) across the band — **not** swept/tuned, and **not** a spectrum dominated by 1–2 spikes.

**Why we ask:** at **500 µs**, a typical commercial fiber agitator ceiling is ~10 kHz (100 µs period), so mechanical scrambling yields at most **~5 patterns** in one exposure; we will still use an agitator to multiply averaging, but **instantaneous spectral diversity is the main mechanism** we want.

---

## 1) Target use case (requirements)

We need **widefield epi** illumination (**not** TIRF, **not** HILO) on a **Nikon Ti2-E** with a **100× high-NA oil** objective.

| Item | 640-ish line (most demanding) | Other lines (nice-to-have) |
|---|---:|---:|
| Wavelengths | 635–655 nm class | 405 / 488 / 561 nm |
| ROI at sample | ~30×30 µm² | ~30×30 µm² |
| Power density at sample | **10–30 kW/cm²** | ≥3 kW/cm² |
| Exposure time | **≤500 µs** | ≥5 ms OK |
| Modulation | TTL and/or analog; clean 500 µs step | TTL and/or analog |
| Speckle goal | no visible speckle at sample plane | low speckle acceptable |

**Delivery concept:** square step-index MMF, **200–400 µm** core (target 400 µm), **higher NA preferred** (e.g. ~0.39 if available; ~0.22 is also common), **~3 m** length; SMA905 or free-space coupling (we can handle coupling).

---

## 2) What we want measured / sent back (most useful data)

If you have (or can easily take) any of the below, it would be extremely helpful — especially **at operating currents/powers relevant to our use case**.

### A) Spectrum (instantaneous)

- optical spectrum analyzer (OSA) traces for the 635–655 nm source at several drive currents / output powers
- confirmation the spectrum is **simultaneously multi-spike** (not scanned/swept)
- any note on whether the spectrum is “many similar spikes” vs “1–2 dominant spikes”

### B) Modulation / pulse fidelity

- TTL + analog modulation specs, and ideally **scope traces** for a **500 µs square pulse**
  - rise/fall time
  - overshoot / ringing
  - timing jitter / repeatability

### C) Speckle / homogeneity at **≤500 µs**

We care about two planes, ideally measured at **≤500 µs** exposure (so we can rule out “long-exposure-only” averaging mechanisms):

1) **MMF exit face near-field** (camera image of the fiber face)
2) **Sample plane / image plane** after the objective (widefield uniformity over ~30×30 µm²)

Helpful outputs include representative images (near-field and/or sample plane) and any quantitative speckle metric (e.g. speckle contrast).

### D) “Multiple diodes combined” option

If combining multiple nearby diodes (e.g. 635/640/645/650/655) is the best path to a broader instantaneous spectrum, we would welcome guidance or an integrated offering (ideally combined into one MMF output).

---

## 3) Published precedents (why we think this direction is plausible)

We are aware that many published “low-cost / diode-based” illumination systems still use a fiber shaker or other scrambler, but they provide useful reference points:

| Reference | What it demonstrates (relevant bits) |
|---|---|
| Almahayni et al., *J. Phys. Chem. A* (2023), DOI: 10.1021/acs.jpca.3c01351 | A practical super-resolution illumination build using a **square-core MM fiber** and a **fiber shaker**; their documentation includes a 638 nm diode spec showing a **few-nm-class linewidth** (2 nm typical, ≤5 nm) |
| Schröder et al., *Biomed. Opt. Express* (2020), DOI: 10.1364/BOE.380815 | A cost-efficient laser engine with **square MMF + agitation**, with speckle-contrast characterization vs exposure time down to the sub-ms regime |
| Kwakwa et al., *J. Biophotonics* (2016), DOI: 10.1002/jbio.201500324 | Shows that increasing MMF core size (up to **400 µm**) improves illumination uniformity, and that a vibrating despeckler improves it further |

Our goal differs mainly in timescale: we need good uniformity at **≤500 µs**, and we are asking whether going to a **much broader instantaneous multi-spike spectrum** can reduce speckle even when mechanical agitation can only contribute a small number of patterns per exposure.

---

## 4) Integration notes (so we don’t talk past each other)

- We need **widefield epi** illumination (uniform ROI), not TIRF and not HILO.
- We are open to critical illumination (imaging the fiber face) or a Köhler-like relay, but **my current understanding is that Köhler-like is preferred** for robustness/uniformity — I’m **not** certain, and I’d value your advice.
- We can handle coupling optics and fiber integration on our side; we mainly need the right source + guidance on what to measure.

---

## 5) Procurement constraints

- The most important item is the **laser source** (broad instantaneous spectrum + fast modulation).
- If you offer an integrated **engine** (especially modular / user-upgradable to add wavelengths later to a common MMF output), that is preferred.

---

## Appendix: simple system diagram (measurement points highlighted)

![](../figures/wide_linewidth_mmf_widefield_system.svg)
