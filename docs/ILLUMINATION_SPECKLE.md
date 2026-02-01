# Illumination design notes: speckle, stops, and FP/FN risk

This document collects (and stabilizes) the illumination-design context we have built so far for
**single-protein widefield imaging** with:

- short exposures (e.g. **5 ms**),
- a small illuminated ROI (e.g. **30 µm × 30 µm**),
- high irradiance (e.g. **kW/cm²-scale**),
- and a desire to understand how excitation nonuniformity affects **Slice0 spot detection**
  false positives / false negatives.

The companion notebook is:

- `notebooks/05_excitation_speckle_fpfn_proxy.py`

It produces:
1) excitation-only field views + distributions, and  
2) a sparse-emitter simulation + confusion-matrix-style FP/FN summary + u0 ECDF plots.

---

## 1) Two independent “design knobs”: field stop vs pupil fill

In microscope widefield illumination there are **two distinct sets of conjugate planes**:

### Field-conjugate (image) planes  → ROI size and shape
A **field stop** placed in a plane conjugate to the sample is imaged onto the sample.  
This sets the **illuminated area** (e.g. a square 30 µm × 30 µm).

### Pupil-conjugate planes (objective back focal plane, BFP) → illumination NA and speckle grain
An **aperture stop** at a pupil-conjugate plane (often the **objective BFP**) controls the
**angular spectrum** of illumination at the sample, i.e. the **effective illumination NA**,
\(\mathrm{NA}_{\mathrm{illum}}\).

These two knobs are often conflated. In this repo’s simulations we treat them separately:

- ROI size: set by a **square field stop** (ideal 30 µm × 30 µm target).
- Speckle grain size + edge sharpness: set by **\(\mathrm{NA}_{\mathrm{illum}}\)** via the pupil.

---

## 2) “Which NA is which?” (quick dictionary)

You will see several NA-like quantities in discussion. Here is a consistent naming set:

1) **\(\mathrm{NA}_{\mathrm{obj}}\)**  
   The objective’s collection / focusing NA (e.g. 1.40 or 1.45 oil).
   This sets the best-case emission PSF and the *maximum possible* excitation cone.

2) **\(\mathrm{NA}_{\mathrm{illum}}\)**  
   The **effective excitation NA at the sample**, determined by how much of the objective pupil
   you actually fill with the illumination beam. If the beam underfills the pupil, then  
   \[
   \mathrm{NA}_{\mathrm{illum}} \approx \rho\,\mathrm{NA}_{\mathrm{obj}},\qquad
   \rho \equiv \frac{D_{\mathrm{beam@BFP}}}{D_{\mathrm{pupil}}}.
   \]
   (\(D_{\mathrm{pupil}}\) depends on the objective design; use a BFP image if you want the true ratio.)

3) **\(\mathrm{NA}_{\mathrm{fiber}}\)**  
   The acceptance NA of the multimode fiber (e.g. 0.22). This describes the **guided mode cone**
   of the fiber itself. It does *not* equal \(\mathrm{NA}_{\mathrm{illum}}\) unless your relay optics
   map the full fiber far-field into the objective pupil.

4) (Sometimes) **source divergence / M²-implied divergence**  
   For a laser beam with beam quality \(M^2\), the far-field divergence is larger than a diffraction-limited
   Gaussian. This can matter *only insofar as it changes* pupil fill and/or effective spatial coherence.

---

## 3) Speckle: what we model and why “random phase” is the right primitive

### 3.1 Physical origin (in one paragraph)
Speckle is an interference phenomenon: at any observation point \(\mathbf{r}\),
the complex field can be written as a superposition of many partial waves / modes,
\[
U(\mathbf{r}) = \sum_{m=1}^{M} a_m(\mathbf{r})\,e^{i\phi_m}.
\]
When there are many contributions with effectively random phases \(\phi_m\), the central-limit theorem
drives \(U\) toward a complex circular Gaussian random variable, and the intensity
\(I = |U|^2\) follows the familiar fully developed speckle statistics (negative exponential for a single
coherent realization).

**That is why “random phase” is not a hack**—it is the simplest correct way to represent the net effect
of many modes with unknown phases.

### 3.2 Coherent vs incoherent vs “speckle-averaged”
These words refer to *how intensities are combined*:

- **Coherent (single realization):** one complex field \(U\), one intensity \(|U|^2\).  
  Highest speckle contrast.

- **Incoherent sum:** sum intensities of mutually incoherent modes/wavelengths/polarizations:  
  \[
  I = \sum_{k=1}^{N} |U_k|^2.
  \]
  Speckle contrast decreases approximately as \(C \sim 1/\sqrt{N}\) when the realizations are independent.

- **Speckle-averaged (time averaged):** camera integrates changing speckle within exposure \(\tau\):  
  \[
  I_{\tau}(\mathbf{r}) = \frac{1}{N}\sum_{n=1}^{N} I_n(\mathbf{r}),
  \qquad N \approx f_{\mathrm{scr}}\,\tau.
  \]
  Mathematically it behaves like an incoherent average if successive patterns are independent.

**Key point:** time averaging and incoherent summation both reduce contrast; the difference is *what*
creates independent realizations.

---

## 4) Two “scales” you care about for single-molecule work

Your false-positive / false-negative behavior is driven by two different tail mechanisms:

### 4.1 Pixel-scale tails (local hotspots)
A single molecule “sees” the excitation intensity at its location. So the pixel-scale (or sub-PSF) field
inhomogeneity affects the distribution of emitter brightness and thus the u0 distribution.

### 4.2 PSF-scale tails (illumination edge spillover)
Even if the field stop is a sharp square, the sample sees a blurred edge because the illumination system
has finite \(\mathrm{NA}_{\mathrm{illum}}\). A useful scale estimate is:
\[
\Delta x_{\mathrm{edge}} \sim \mathcal{O}\left(\frac{\lambda}{\mathrm{NA}_{\mathrm{illum}}}\right).
\]
This is why underfilling the pupil (small \(\mathrm{NA}_{\mathrm{illum}}\)) simultaneously:
- increases speckle grain size, and
- makes the “square” edge less sharp.

The notebook explicitly shows both pixel-level ROIs and an edge-linecut view.

---

## 5) Speckle grain size (and why it can’t be “sub-pixel” for 65 nm sampling)

A common order-of-magnitude estimate for speckle grain size in the sample plane is:
\[
\Delta x_{\mathrm{speckle}} \approx \frac{\lambda}{2\,\mathrm{NA}_{\mathrm{illum}}}.
\]

With \(\lambda = 640\,\mathrm{nm}\) and sample-plane sampling \(\Delta x = 65\,\mathrm{nm/px}\):
- If \(\mathrm{NA}_{\mathrm{illum}} = 0.05\):  
  \(\Delta x_{\mathrm{speckle}} \approx 6.4\,\mathrm{\mu m} \approx 98\,\mathrm{px}\) (very large grains).
- If \(\mathrm{NA}_{\mathrm{illum}} = 0.20\):  
  \(\Delta x_{\mathrm{speckle}} \approx 1.6\,\mathrm{\mu m} \approx 25\,\mathrm{px}\).
- To get \(\Delta x_{\mathrm{speckle}}\lesssim 1\,\mathrm{px}\) would require \(\mathrm{NA}_{\mathrm{illum}}\gtrsim 5\),
  which is impossible.

So the realistic objective is not “sub-pixel speckle,” but rather **low speckle contrast** within the ROI
at the relevant exposure time.

---

## 6) Where does M² fit?

\(M^2\) is a beam-propagation (second-moment) metric. It does **not** uniquely determine speckle,
because speckle is controlled by **spatial coherence** at the sample and by how many *independent*
realizations are averaged during an exposure.

However, in design exploration it is still useful as a *proxy knob*:

- Larger \(M^2\) often implies more transverse structure / larger angular spread.
- If those components are mutually incoherent (or rapidly decorrelated by a scrambler + mode coupling),
  effective speckle contrast can drop.

**Caution:** the mapping \(M^2 \to N_{\mathrm{eff}}\) is *not universal*.  
The notebook treats \(M^2\) as an adjustable proxy for additional “incoherent diversity.”
You should calibrate this using a simple experiment:

1. Fix \(\mathrm{NA}_{\mathrm{illum}}\) (BFP fill) and ROI size.
2. Record repeated short-exposure frames.
3. Measure speckle contrast \(C\) in the inner ROI.
4. Back out an empirical \(N_{\mathrm{eff}} \approx 1/C^2\) for your actual hardware.

---

## 7) Free-space Gaussian (no fiber) vs multimode fiber + scrambler + field stop

### Free-space near-TEM00 Gaussian injection (no fiber)
**Pros**
- No fiber speckle / mode noise.
- Fewer components; can be high-throughput.
- Can be very stable if the optomechanics are rigid and clean.

**Cons**
- Gaussian nonuniformity across a 30 µm ROI unless you oversize the beam (wastes power).
- Any attempt to “hard crop” a coherent Gaussian with a square stop can introduce diffraction ringing
  (fringes), which can become its own FP/FN risk near edges.
- Alignment drift is usually worse than fiber delivery.

### Multimode fiber + scrambler + square field stop
**Pros**
- Convenient delivery to the microscope; robust beam pointing at the back port.
- Field stop makes it straightforward to define a crisp ROI (shape set geometrically).
- With sufficient averaging (scrambler, mode mixing), can approximate a top-hat ROI with low speckle contrast.

**Cons**
- Without enough averaging at \(\tau=5\,\mathrm{ms}\), residual speckle can dominate local background variations.
- Mechanical scramblers have finite decorrelation rates; “independent pattern per cycle” is approximate.
- Extra components introduce loss, alignment, and potential autofluorescence (wavelength dependent).

The notebook is designed to compare these tradeoffs *quantitatively* in terms of spot detection metrics.

---

## 8) What do other labs do (from `y24m06d25_LASERS_NIKON_0.docx`)?

The setups collected in that document predominantly describe **TIRF/HILO-style injection**
(focusing beams into the objective BFP, often near the edge for inclined illumination).
Several explicitly mention **fiber-coupled** delivery, and at least one explicitly uses a
**single-mode fiber**. The document does not (in the excerpts we have) explicitly describe a
multimode-fiber + scrambler “speckle reducer” widefield approach.

---

## 9) Running the notebook

From the repo root:

```bash
conda activate bioimg-pipeline
jupyter lab
```

Open:

- `notebooks/05_excitation_speckle_fpfn_proxy.py`

Suggested first run:
- keep defaults (\(\mathrm{NA}_{\mathrm{illum}}\approx 0.05\), scrambler 10 kHz, 5 ms)
- then sweep **\(\mathrm{NA}_{\mathrm{illum}}\)** and **scrambler frequency** and watch:
  - inner-ROI intensity distribution,
  - speckle contrast estimate,
  - FP/FN behavior under Slice0.

---

## 10) Scope and limitations

This model is intentionally scalar and 2D. It does **not** include:
- vector polarization effects,
- objective aberrations / coverslip mismatch,
- fluorescence saturation / photophysics,
- coherent interference from reflections (etalon fringes),
- true fiber mode solver physics.

But it is still highly useful as a *design intuition tool* because it forces you to keep the conjugate planes straight,
ties \(\mathrm{NA}_{\mathrm{illum}}\) to speckle/edge scales, and connects field statistics to spot detection outcomes.
