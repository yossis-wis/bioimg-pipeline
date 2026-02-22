# Draft email: multi-vendor inquiry (wide-linewidth 640-ish lasers for speckle-reduced **widefield epi**)

Vendors of interest (examples): Oxxius, 89 North, CNI, Thorlabs, Omicron, Lasertack, Ushio.

---

## Email draft (copy/paste)

**To:** [sales / applications contact at COMPANY]  
**Subject:** Question: **broadest instantaneous linewidth** 640-ish source for speckle-reduced **widefield epi** (500 µs) via MMF

Hi [Name],

I’m looking for your advice (and, if possible, any measured data) on a **speckle-reduction** approach for **widefield epi** fluorescence microscopy on a **Nikon Ti2-E** (**not** TIRF, **not** HILO).

**Main question:** what is the **broadest instantaneous spectrum** you can offer in the **635–655 nm** class?  
By “broad,” I specifically mean **multi-longitudinal-mode / multi-spike simultaneously present** (ideally ~100 spikes or more), with spike powers roughly comparable (or smoothly varying / Gaussian-like) across the band — **not** a swept/tuned source, and **not** a spectrum dominated by 1–2 spikes.

**Target use case (most demanding line):**
- Field at sample: **30×30 µm²**, 100× high-NA objective (oil)
- Power density at sample: **10–30 kW/cm²** (≈ 90–270 mW at sample)
- Timing: **500 µs** flat pulses (fast TTL and/or analog; rise/fall and jitter ≪ 500 µs)
- Goal: **no visible speckle** at the sample plane at 500 µs

**Speckle approach:** deliver 640-ish light through a large-core **square step-index MMF** (target **400 µm** core, **higher NA preferred**; e.g. **NA ≈ 0.39** if available; ~3 m length; SMA905 or free-space coupling). We may also use a commercial **~10 kHz fiber agitator**: at 500 µs it yields at most ~5 patterns, so **spectral diversity is the main homogenization mechanism**, but we will still use the agitator to boost averaging further. For our other lines (405/488/561 nm), exposures **≥5 ms** are fine (so ~50 patterns at 10 kHz), and the agitator should be sufficient (possibly unnecessary if those lines are also broad).

### Questions (quick checklist)
1) **640-ish linewidth:** what is the **broadest-linewidth** 635–655 nm source you have available (module/diode/engine), and can you share measured **instantaneous** spectra (OSA traces) at relevant currents/powers?  
2) **Multiple-diode option:** if the best route to a broad spectrum is combining multiple nearby diodes (e.g. 635/640/645/650/655), is that something you can supply (or recommend) as a robust, combined output (ideally into one MMF)?  
3) **Modulation:** can you share TTL + analog step response for a **500 µs square pulse** (bandwidth, rise/fall, overshoot/ringing, jitter)? Scope traces welcome.  
4) **Homogeneity measurements (if available):** have you measured or can you comment on speckle / uniformity:
   - at the **MMF exit face (near-field)**, and/or
   - after imaging through an objective to an **image/sample plane** (widefield)?
5) **Fiber guidance:** do you agree that square step-index, large-core, higher-NA MMF + a few meters is a good direction? Any recommended core size/NA/length ranges?  
6) **Sanity check:** is there a flaw in my conception (a reason this “instantaneous multi-spike + MMF” approach would fail to remove speckle at 500 µs)?

Procurement note: I primarily need the **laser source(s)**; we can handle coupling optics and fiber. If you offer an integrated **engine** (especially modular / user-upgradable to add wavelengths later to a common MMF output), that would be preferred.

I also have a short **1–2 page supporting note** with diagrams and more explicit measurement targets, if you’d like it forwarded to an applications engineer.

Best regards,  
[Your name]  
[Institution]  
[Phone / email]
