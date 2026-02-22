# Draft email: multi-vendor inquiry (wide-linewidth 640-ish lasers for speckle-reduced **widefield epi**) — concise

Vendors of interest (examples): Oxxius, 89 North, CNI, Thorlabs, Omicron, Lasertack, Ushio.

---

## Email draft (copy/paste) — ≤10 sentences

**To:** [sales / applications contact at COMPANY]  
**Subject:** Main question: **broadest instantaneous linewidth** 635–655 nm source for speckle-free **widefield epi** at 500 µs

Hi [Name],

I’m trying to get **speckle-free widefield epi** illumination (**not** TIRF, **not** HILO) on a Nikon Ti2-E at **500 µs** exposures, and I’m looking for the best 640-ish laser option you have.

**Main question:** in the **635–655 nm** class, what **multimode** source has your **largest instantaneous spectral linewidth (FWHM)** (ideally centered ~646–650 nm and spanning as much of 635–655 nm as possible)?

By “broad,” I mean **simultaneously present** many spectral spikes (order 100+), with roughly comparable / Gaussian-like spike powers — because more simultaneous spikes should average out more independent speckle patterns within one exposure — **not** swept/tuned, and **not** a spectrum dominated by 1–2 spikes.

Quick spec summary:

| Item | Target |
|---|---|
| Geometry | **Widefield epi** (uniform ROI), **not** TIRF, **not** HILO |
| Microscope | Nikon Ti2-E, 100× high-NA (oil) |
| ROI at sample | ~30×30 µm² |
| 640-ish line | 635–655 nm class; **500 µs** pulses; **10–30 kW/cm²** at sample |
| Delivery | square step-index MMF, 200–400 µm core (target 400 µm), high NA preferred (e.g. ~0.39), ~3 m |
| Other lines | 405 / 488 / 561 nm; exposures ≥5 ms OK |

Simple system diagram:

![](../figures/wide_linewidth_mmf_widefield_system.svg)

We plan to deliver this through a **square step-index multimode fiber (MMF)** (target **400 µm** core, **higher NA preferred** e.g. ~0.39 if available, ~3 m long; we can handle coupling), optionally plus a ~10 kHz fiber agitator (useful but only ≤5 patterns in 500 µs, so **spectral diversity must dominate**).

For context, similar “low-cost diode + square MMF (+/− agitation)” illumination approaches appear in super-resolution setups (e.g. Almahayni et al., *J. Phys. Chem. A* 2023, DOI: 10.1021/acs.jpca.3c01351), but we need uniformity at **≤500 µs**.

Could you help with any of the following (data or a brief yes/no is fine)?

1) Your **broadest** 635–655 nm offering and measured **instantaneous optical spectrum analyzer (OSA) spectrum** vs current/power
2) TTL (digital) + analog **500 µs step response** (scope trace welcome)
3) **Speckle / uniformity** at **≤500 µs** at the **MMF exit face** and/or **sample plane** (widefield; ~30×30 µm²)
4) If combining multiple nearby diodes (635/640/645/650/655) into one MMF output is a better path to broad spectrum, whether you can supply/recommend it
5) Sanity-check: any reason a broad-linewidth multimode source + square step-index MMF would *not* homogenize at ≤500 µs

Procurement-wise, I mainly need the **laser(s)**, but an integrated, **user-upgradable multi-line engine** (common MMF output) would be preferred.

If helpful, I can forward a short 1–2 page supporting note with a diagram + an explicit measurement checklist for your applications engineer.

Best regards,  
[Your name]  
[Institution]  
[Phone / email]
