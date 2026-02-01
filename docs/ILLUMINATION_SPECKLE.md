# Illumination design notes: speckle, stops, and FP/FN risk

> **GitHub math note:** GitHub renders LaTeX math only when it is wrapped in `$...$` (inline) or `$$...$$` (display).
> Do **not** use `\(...\)` or `\[...\]` in this repo’s `.md` docs.

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

1. excitation-only field views + distributions, and  
2. a sparse-emitter simulation + confusion-matrix-style FP/FN summary + $u_0$ ECDF plots.

---

## 1) Two independent “design knobs”: field stop vs pupil fill

In widefield illumination there are **two distinct sets of conjugate planes**.

### Field-conjugate (image) planes → ROI size and shape

A **field stop** placed in a plane conjugate to the sample is imaged onto the sample.
This sets the **illuminated area** (e.g. a square $30\,\mu\mathrm{m} \times 30\,\mu\mathrm{m}$).

### Pupil-conjugate planes (objective back focal plane, BFP) → illumination NA and spatial frequencies

An **aperture stop** at a pupil-conjugate plane (often the objective **BFP**) controls the **angular spectrum**
of illumination at the sample, i.e. the **effective illumination NA**, $\mathrm{NA}_{\mathrm{illum}}$.

These two knobs are often conflated. In this repo’s simulations we treat them separately:

- ROI size: set by a **square field stop** (ideal $30\,\mu\mathrm{m} \times 30\,\mu\mathrm{m}$ target).
- Speckle correlation length + edge sharpness: set by $\mathrm{NA}_{\mathrm{illum}}$ via the pupil.

---

## 2) “Which NA is which?” (quick dictionary)

You will see several NA-like quantities in discussion. Here is a consistent naming set:

1. **$\mathrm{NA}_{\mathrm{obj}}$**  
   The objective’s collection/focusing NA (e.g. 1.40 or 1.45 oil).
   This sets the best-case **emission PSF** and the *maximum possible* excitation cone.

2. **$\mathrm{NA}_{\mathrm{illum}}$**  
   The **effective excitation NA at the sample**, determined by how much of the objective pupil you fill
   with the illumination beam. If the beam underfills the pupil, then (paraxial) $\mathrm{NA}_{\mathrm{illum}} \approx \rho\,\mathrm{NA}_{\mathrm{obj}}$, with $\rho \equiv D_{\mathrm{beam@BFP}}/D_{\mathrm{pupil}}$.\n   ($D_{\mathrm{pupil}}$ depends on the objective design; use a BFP image if you want the true ratio.)

3. **$\mathrm{NA}_{\mathrm{fiber}}$**  
   The acceptance NA of the multimode fiber (e.g. 0.22). This describes the **guided mode cone**
   of the fiber itself. It does *not* equal $\mathrm{NA}_{\mathrm{illum}}$ unless your relay optics map the
   fiber far-field into the objective pupil.

4. (Sometimes) **source divergence / $M^2$-implied divergence**  
   For a laser beam with beam quality $M^2$, the far-field divergence is larger than a diffraction-limited
   Gaussian. This can matter only insofar as it changes pupil fill and/or the effective spatial coherence
   at the sample.

---

## 3) Speckle: what we model and why “random phase” is the right primitive

### 3.1 Physical origin (one paragraph)

Speckle is an interference phenomenon: at any observation point $\mathbf{r}$, the complex field can be written
as a superposition of many partial waves / modes,
$$
U(\mathbf{r}) = \sum_{m=1}^{M} a_m(\mathbf{r})\,e^{i\phi_m}.
$$
When there are many contributions with effectively random phases $\phi_m$, the central-limit theorem drives
$U$ toward a complex circular Gaussian random variable, and the intensity $I = |U|^2$ follows the familiar
fully developed speckle statistics (negative exponential for a single coherent realization).

That is why “random phase” is not a hack—it is the simplest correct representation of many modes with unknown phases.

### 3.2 Coherent vs incoherent vs “speckle-averaged” (what those words mean)

These terms describe *how intensities are combined*.

#### Coherent (single realization)

One complex field $U(\mathbf{r})$, one intensity $I(\mathbf{r}) = |U(\mathbf{r})|^2$.  
This has the **highest speckle contrast**.

#### Incoherent sum

Sum intensities of mutually incoherent modes/wavelengths/polarizations:
$$
I(\mathbf{r}) = \sum_{k=1}^{N} |U_k(\mathbf{r})|^2.
$$
If the $U_k$ are independent, speckle contrast decreases approximately as $C \sim 1/\sqrt{N}$.

#### Speckle-averaged (time averaged)

The camera integrates changing speckle during an exposure time $\tau$:
$$
I_{\tau}(\mathbf{r}) = \frac{1}{N}\sum_{n=1}^{N} I_n(\mathbf{r}),\qquad
N \approx f_{\mathrm{scr}}\,\tau .
$$
Mathematically this behaves like an incoherent average *if* successive patterns are independent.

**Key point:** time averaging and incoherent summation both reduce contrast; the difference is *what* creates independent realizations.

### 3.3 “Can I skip the scrambler and just calibrate a static speckle pattern?”

Sometimes—*but it is risky for 5 ms single-molecule work*.

A speckle pattern changes when relative optical phases change by $\sim 1\,\mathrm{rad}$.
For a path-length change $\Delta L$ in glass (index $n$), a phase change of 1 rad corresponds to
$\Delta L \sim \lambda/(2\pi n)$ (tens of nm at $\lambda\approx 640\,\mathrm{nm}$).
That is a very small mechanical/thermal perturbation.

So with a multimode fiber, **even minute bending, vibration, or thermal drift** can decorrelate the pattern.
Without an intentional scrambler you can easily end up in the worst regime: high-contrast speckle *that also drifts*,
creating frame-to-frame multiplicative noise that is hard to flat-field away.

If you want the “static calibratable” strategy, it is usually more realistic with a **single-mode / clean free-space beam**
where the dominant nonuniformity is a smooth Gaussian envelope (and any interference fringes are minimized by good baffling/tilts).

---

## 4) Two spatial-frequency regimes that matter for single-molecule spot detection

Your FP/FN behavior is driven by two different “tail” mechanisms.

### 4.1 PSF-scale structure (dangerous for FP)

If excitation nonuniformity has significant spatial power at the **emission PSF scale**
($\sim 200$–$300\,\mathrm{nm}$ for a 1.4 NA objective at visible wavelengths), it can generate
local maxima that resemble real spots and increase false positives.

This is why it is useful to compare:

- excitation-field structure at **pixel scale** ($65\,\mathrm{nm}$ sampling), and
- structure at the **PSF scale** (after convolution with an emission PSF model).

### 4.2 ROI-scale structure (biases FN and quantification)

If the excitation varies slowly compared to the PSF (e.g. speckle grains of several $\mu\mathrm{m}$),
it mostly acts like a **multiplicative flat-field**. That tends to:

- bias *where* molecules are bright enough to be detected (false negatives in dark regions),
- bias brightness/photobleaching rates across the ROI,
- but is less likely to create PSF-like false positives.

The notebook is built to quantify both regimes (excitation-only + sparse-emitter FP/FN proxy).

---

## 5) Speckle grain size (correlation length) and why it cannot be “sub-pixel” for 65 nm sampling

A common order-of-magnitude estimate for speckle grain size (intensity correlation width) in the sample plane is:
$$
\Delta x_{\mathrm{speckle}} \approx \frac{\lambda}{2\,\mathrm{NA}_{\mathrm{illum}}}.
$$

With $\lambda = 640\,\mathrm{nm}$ and sampling $\Delta x = 65\,\mathrm{nm/px}$:

- If $\mathrm{NA}_{\mathrm{illum}} = 0.05$:  
  $\Delta x_{\mathrm{speckle}} \approx 6.4\,\mu\mathrm{m} \approx 98\,\mathrm{px}$ (very large grains).
- If $\mathrm{NA}_{\mathrm{illum}} = 0.20$:  
  $\Delta x_{\mathrm{speckle}} \approx 1.6\,\mu\mathrm{m} \approx 25\,\mathrm{px}$.

To get $\Delta x_{\mathrm{speckle}} \lesssim 1\,\mathrm{px}$ would require $\mathrm{NA}_{\mathrm{illum}} \gtrsim 5$,
which is impossible.

So the practical objective is not “sub-pixel grains,” but **low speckle contrast** (via averaging) and/or
a correlation length that does **not** inject structure at the emission-PSF scale.

---

## 6) Where does $M^2$ fit?

$M^2$ is a beam-propagation (second-moment) metric. It does **not** uniquely determine speckle, because speckle is controlled by

- spatial coherence at the sample, and
- how many *independent* realizations are averaged during an exposure.

However, in design exploration it is still useful as a *proxy knob*:

- Larger $M^2$ often implies more transverse structure / larger angular spread.
- If those components are mutually incoherent (or rapidly decorrelated by a scrambler + mode coupling),
  effective speckle contrast can drop.

**Caution:** the mapping $M^2 \rightarrow N_{\mathrm{eff}}$ is *not universal*.  
The notebook treats $M^2$ as an adjustable proxy for additional “incoherent diversity.”
You should calibrate this empirically for your actual hardware:

1. Fix $\mathrm{NA}_{\mathrm{illum}}$ (BFP fill) and ROI size.
2. Record repeated short-exposure frames.
3. Measure speckle contrast $C$ in the inner ROI.
4. Estimate $N_{\mathrm{eff}} \approx 1/C^2$.

---

## 7) Free-space Gaussian (no fiber) vs multimode fiber + scrambler + field stop

### Free-space near-TEM00 Gaussian injection (no fiber)

**Pros**

- No fiber-mode speckle / mode noise.
- Fewer components; can be high-throughput.
- Can be very stable if optomechanics are rigid and the beam path is enclosed.

**Cons**

- Gaussian nonuniformity across a $30\,\mu\mathrm{m}$ ROI unless you oversize the beam (wastes power).
- Coherent “hard cropping” (square stop) can introduce diffraction ringing (fringes) unless you image the stop
  incoherently (Köhler-like) or accept soft edges.
- Back reflections can destabilize some laser heads unless you use an isolator and/or careful tilt/wedge management.

### Multimode fiber + scrambler + square field stop

**Pros**

- Convenient delivery to the microscope; robust beam pointing at the back port.
- Field stop makes it straightforward to define ROI shape geometrically.
- With sufficient averaging (scrambler + mode mixing), can approximate a top-hat ROI with low speckle contrast.

**Cons**

- Without enough averaging at $\tau=5\,\mathrm{ms}$, residual speckle can dominate pixel-level nonuniformity.
- Mechanical scramblers have finite decorrelation rates; “independent pattern per cycle” is approximate.
- Extra components introduce loss, alignment complexity, and potential wavelength-dependent behavior.

The notebook is designed to compare these tradeoffs quantitatively in terms of spot detection metrics.

---

## 8) Square fiber vs square field stop: does relative rotation matter?

If the beam at the field stop **overfills** the stop in both dimensions, rotation is usually not important
(the stop defines the ROI).

If you are trying to maximize throughput by matching a square-core image to a square stop, then relative rotation can:

- clip corners (power loss),
- and subtly bias edge uniformity.

Practical rule: oversize the square-core image at the stop by ~10–20% and treat the stop as the ROI-defining element.
