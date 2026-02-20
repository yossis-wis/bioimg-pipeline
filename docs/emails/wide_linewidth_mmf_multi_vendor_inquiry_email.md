# Draft email: multi-vendor inquiry (wide-linewidth lasers + square step-index MMF for speckle-free 640 nm widefield)

Vendors of interest (examples): Oxxius, 89 North, CNI, Thorlabs, Omicron, Lasertack, Ushio.

---

## Email draft (copy/paste)

**To:** [sales / applications contact at COMPANY]  
**Subject:** Inquiry: wide-linewidth (multi-spike) 640 nm excitation + MMF delivery for speckle-free widefield (500 µs) on Nikon Ti2-E

Hi [Name],

I’m looking for an engineer’s perspective (and, if feasible, a quote) on a **low-speckle widefield epi-illumination** approach for fluorescence microscopy.

### What we need at the sample (widefield, not TIRF / not HILO)
- Microscope: **Nikon Eclipse Ti2-E** (inverted widefield)
- Objective: **100× high-NA** (oil)
- Illumination field: **30×30 µm²** at the sample
- **640-ish excitation (most demanding):**
  - power density: **10–30 kW/cm²** over 30×30 µm² (≈ **90–270 mW at the sample**)
  - timing: **clean 500 µs pulses** (TTL and/or analog), with a **flat intensity step** (rise/fall and jitter ≪ 500 µs)
- **Other lines (ideal full system):** 405 / 488 / 561 nm
  - power density goal: **≥3 kW/cm²** over 30×30 µm²
  - exposures: **≥5 ms** are fine

### The approach we want to validate
We are trying to achieve **homogeneous illumination (no speckle / no interference artifacts)** by combining:
1) a **broad spectral linewidth, multi-longitudinal-mode** laser spectrum (many spikes present simultaneously), and
2) delivery through a **large-core square step-index multimode fiber (MMF)** (target: ~200–400 µm core, ~3 m length).

Crucially, for the **640 nm / 500 µs** case we want this to work **without mechanical motion** (no rotating diffuser / no fiber vibrator). For the **≥5 ms** lines we are open to adding a commercial ~**10 kHz fiber agitator** if helpful, but we still prefer the wide-linewidth approach for all wavelengths.

We can source the fiber and do the coupling in-house; the most important part for us is the **laser source (spectrum + modulation)**. That said, an **integrated multi-line system with a common MMF output** (and modular upgrade path) would be ideal.

### Simple diagram of what we mean
![](../figures/wide_linewidth_mmf_widefield_system.svg)

### Questions (what would you recommend / what can you provide?)
1) Do you have a **640-ish module** (or diode-based source) that is explicitly **multi-longitudinal-mode / broad linewidth** (widest linewidth available; many spikes simultaneously present, not swept)?
2) Can you share any **measured spectra** (OSA traces) at operating powers/currents?
3) What are the **modulation specs** (TTL and analog): rise/fall time, bandwidth, and behavior for a **500 µs step** (overshoot/ringing/jitter)?
4) Have you measured or can you comment on **illumination homogeneity / speckle**:
   - at the **MMF exit face near-field**, and/or
   - after imaging through an objective to an **image plane / sample plane**?
5) Fiber advice: for speckle reduction, do you agree that **square, step-index, large core (200–400 µm)** and **a few meters** of fiber is a good direction? What length do you consider realistic?
6) If the best way to widen the spectrum is to **combine multiple nearby diodes** (e.g. 635 / 640 / 645 / 650 / 655 nm, each a few nm wide), is that something you can supply as an integrated solution (or recommend how to do it robustly)?

Practical note: this will be used as a **departmental resource** by researchers without an optics background, so robustness and repeatability matter.

If it’s easier, I’m happy to provide more details (coupling geometry, available ports on the Ti2-E, etc.), but I wanted to first check if this approach aligns with what you’ve seen work in practice.

Best regards,  
[Your name]  
[Institution]  
[Phone / email]
