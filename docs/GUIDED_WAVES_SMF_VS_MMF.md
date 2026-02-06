# Guided-wave primer for this repo
## Single‑mode vs multimode fiber (SMF vs MMF), with an MMF‑speckle focus

**Draft 3 (2026‑02‑06)**

This is a guided‑wave / fiber‑optics “chapter‑style” primer written specifically to support the two
illumination‑delivery approaches in this repo:

- **Approach A (SMF per wavelength):** stable spatial mode (near‑Gaussian), then combine in free space.
- **Approach B (MMF for multiple wavelengths):** many spatial modes (speckle) + **averaging** (linewidth, scrambler, etc.).

Printing note: everything is intended to print cleanly on a black‑and‑white laser printer.
Figures are **original** (drawn for this repo) and use only black strokes.

---

## 0) How to use this document

If you only read 3 pages, read:

1) **Section 1:** the two approaches and what problem we’re actually solving.  
2) **Section 5:** the **V‑number** and “single‑mode cutoff”.  
3) **Section 10–13:** why MMF speckle depends on **coherence length** and **path‑length spread**.

If you want one “mental model” to keep in your head:

> **MMF speckle exists because the MMF output field is a coherent sum of many modes.  
> Anything that destroys the *mutual coherence* between many modal contributions (spectral width, fast scrambling, polarization diversity, angle diversity) reduces speckle contrast in the camera exposure.**

---

## 1) The two fiber approaches in this repo (one picture + one sentence each)

### Approach A: single‑mode fiber per wavelength

```
[Laser 640]--(SMF)->(collimator)--\
                                  +--> [combine in free space] --> [relay to objective]
[Laser 488]--(SMF)->(collimator)--/
```

One sentence: **the output spatial mode is predictable** (approximately Gaussian), so the illumination is
smooth and stable.

### Approach B: wide‑linewidth sources into one multimode fiber

```
[Laser 640 free-space]--\
                         +--> [combine] --> [MMF] --> [scrambler] --> [collimate + stop] --> [objective]
[Laser 488 free-space]--/
```

One sentence: **the MMF supports many modes**, so the output is speckle unless the camera integrates an
average over many *effectively independent* speckle realizations.

---

## 2) Symbol legend (keep nearby)

This is a “cheat sheet” for symbols used repeatedly.

### Geometry and indices

- $`a`$ : **core radius** (m).  
  Example: $`a=1.5\thinspace \mu\mathrm{m}`$ (SMF), or $`a=200\thinspace \mu\mathrm{m}`$ (400‑µm‑core MMF).

- $`n_1`$ : **core refractive index** (unitless).  
  Representative visible silica: $`n_1\approx 1.46`$.

- $`n_2`$ : **cladding refractive index** (unitless), with $`n_2 < n_1`$.

- $`\Delta`$ : **fractional index contrast** (unitless). A common weak‑guidance definition is
  ```math
  \Delta \equiv \frac{n_1^2-n_2^2}{2n_1^2}.
  ```

### Wavelength, frequency, wavenumbers

- $`\lambda`$ : **vacuum wavelength** (m). (We use vacuum wavelength unless stated.)
- $`k_0`$ : **vacuum wavenumber** (rad/m):
  ```math
  k_0 \equiv \frac{2\pi}{\lambda}.
  ```
- $`\omega`$ : **angular frequency** (rad/s), $`\omega = 2\pi f`$.

### Fiber acceptance and “how many modes?”

- $`\mathrm{NA}`$ : numerical aperture (unitless). For step‑index in air,
  ```math
  \mathrm{NA} = \sqrt{n_1^2-n_2^2} \approx n_1\sqrt{2\Delta}.
  ```

- $`V`$ : **V‑number** (a dimensionless “how many modes?” predictor):
  ```math
  V \equiv \frac{2\pi a}{\lambda}\thinspace \mathrm{NA}.
  ```

- $`M`$ : approximate number of guided spatial modes (unitless count).

### MMF speckle + coherence

- $`\Delta\mathrm{OPL}`$ : **optical path‑length spread** across guided contributions (m).
- $`\Delta\tau`$ : corresponding **group‑delay spread** (s), approximately $`\Delta\tau \approx \Delta\mathrm{OPL}/c`$.
- $`L_c`$ : coherence length (m).
- $`\Delta\lambda_c`$ : spectral decorrelation width (nm or m, context).
- $`N_{\mathrm{eff}}`$ : effective number of independent “looks” averaged in one exposure.

---

## 3) What is a “guided mode” (wave view, not ray view)

Ray optics can tell you whether light is trapped by total internal reflection.
Wave optics tells you what the *field* looks like.

A standard guided‑mode ansatz is:

```math
\mathbf{E}(x,y,z,t) = \Re\Big\{\mathbf{e}(x,y)\thinspace \exp\big(i\beta z - i\omega t\big)\Big\}.
```

Read it slowly:

- $`\mathbf{E}`$ is the **real** electric field (what nature has).
- $`\mathbf{e}(x,y)`$ is the **transverse mode profile** (complex amplitude across the cross‑section).
- $`\exp(i\beta z)`$ means “as you go forward in $z$, the phase advances at rate $`\beta`$”.
- $`\beta`$ is the **propagation constant** (units rad/m).
- $`\exp(-i\omega t)`$ is the usual time oscillation.

A **guided mode** is a specific transverse pattern $`\mathbf{e}_m(x,y)`$ that satisfies Maxwell’s equations
*and* boundary conditions at the core–cladding interface. Each guided mode has its own $`\beta_m`$.

### Why do only certain modes exist?

The Oron guided‑waves notes develop this first for a **slab waveguide**:
a plane wave bouncing between interfaces must reproduce itself after a round trip.
That “self‑consistency” forces certain transverse phase conditions → **discrete allowed angles** → discrete $`\beta`$.

The same physical idea survives in a cylindrical fiber, but the math involves Bessel functions.

---

## 4) Step‑index vs graded‑index profiles (this matters for Approach B)

### 4.1 Step‑index profile

A step‑index fiber has:

```math
n(r) =
\begin{cases}
 n_1, & r < a \\
 n_2, & r > a.
\end{cases}
```

So the index jumps at $`r=a`$.

### 4.2 Graded‑index (GI) profile (common in “homogenizing” fibers)

A commonly used graded‑index model is a power‑law:

```math
n(r) = n_1\sqrt{1-2\Delta\left(\frac{r}{a}\right)^\alpha}, \qquad r \le a,
```

with $`\alpha > 0`$:

- $`\alpha \to \infty`$ (or “very large”) approaches the step‑index limit.
- $`\alpha=2`$ gives a **parabolic index** profile (a classic GI design).

**Key qualitative consequence (GI fiber):** rays that wander far from the axis see a *lower* refractive index,
so they move faster; this can partially “equalize” arrival times and reduce intermodal dispersion.

### 4.3 Illustration: why GI reduces intermodal delay (hard to feel from equations alone)

![Ray paths in step-index vs graded-index fiber](figures/ray_paths_step_vs_grin.svg)

If you print without SVG support, the same idea in ASCII:

```
Step-index (SI):           Graded-index (GI / parabolic):
n = n1 inside core         n(r) decreases with r

  |---------|                |---------|
  |  /\  /\ |                |   ~~~   |
  | /  \/  \|                |  ~   ~  |
  |/        \                | ~     ~ |
  |----------|                |---------|

High-angle path is longer    High-angle path is longer,
*and* speed is similar       but speed is higher near edge
→ larger delay spread        → delay spread reduced
```

For Approach B this is not a footnote: GI can be “good” for telecom but “bad” for *spectral speckle averaging*,
because reduced delay spread means fewer independent spectral looks (Section 13).

---

## 5) Numerical aperture (NA): define it, then compute it once

### 5.1 NA from acceptance angle (ray picture)

By definition:

```math
\mathrm{NA} \equiv n_0\sin\theta_{\max},
```

where:

- $`n_0`$ is the external medium index (≈1 for air).
- $`\theta_{\max}`$ is the maximum external half‑angle that can be accepted (in the ray sense).

So in air, $`\mathrm{NA}\approx \sin\theta_{\max}`$.

### 5.2 NA from indices (step‑index fiber)

For a step‑index fiber in air, a standard result is:

```math
\mathrm{NA} = \sqrt{n_1^2-n_2^2}.
```

**Toy numeric example (indices → NA):**

Take:

- $`n_1=1.460`$
- $`n_2=1.455`$

Step by step:

1) Square them  
   $`n_1^2 = 1.460^2 \approx 2.1316`$  
   $`n_2^2 = 1.455^2 \approx 2.1170`$

2) Subtract  
   $`n_1^2-n_2^2 \approx 0.0146`$

3) Square‑root  
   $`\mathrm{NA} \approx \sqrt{0.0146} \approx 0.121`$

So:

```math
\theta_{\max} \approx \arcsin(0.121) \approx 6.9^\circ.
```

That “small acceptance angle” is why small‑angle approximations are everywhere in fiber estimates.

---

## 6) The V‑number: one dimensionless number that predicts “single‑mode vs multimode”

### 6.1 Definition

Define:

```math
V \equiv \frac{2\pi a}{\lambda}\thinspace \mathrm{NA}.
```

Every factor is intuitive:

- $2\pi/\lambda$ sets a “spatial frequency scale” (how fine features can be).
- $a$ sets the transverse size of the waveguide.
- NA sets how strongly the guide can confine transverse components.

### 6.2 Single‑mode cutoff (step‑index cylindrical fiber)

A classic result (Agrawal; Hecht; most fiber texts) is:

> A step‑index cylindrical fiber is strictly single‑mode if $V < 2.405$.

The number 2.405 is the first zero of the Bessel function $`J_0`$.

### 6.3 Toy calculation: “is my fiber single‑mode at 488 and 640?”

Assume a “visible SMF” with:

- $`a = 1.5\thinspace \mu\mathrm{m} = 1.5\times 10^{-6}\thinspace \mathrm{m}`$
- $`\mathrm{NA}=0.12`$

Compute for each wavelength.

#### For $488\thinspace \mathrm{nm}`$

1) Convert  
   $`\lambda = 488\thinspace \mathrm{nm} = 488\times 10^{-9}\thinspace \mathrm{m}`$

2) Insert into the formula  
   ```math
   \begin{aligned}
   V_{488}
   &= \frac{2\pi (1.5\times 10^{-6})}{488\times 10^{-9}}(0.12) \\
   &= 2\pi\left(\frac{1.5}{488}\times 10^{3}\right)(0.12) \\
   &\approx 2\pi(3.074)(0.12) \\
   &\approx 2.32.
   \end{aligned}
   ```

3) Compare to 2.405  
   $`2.32 < 2.405`$ → **single‑mode** (barely).

#### For $640\thinspace \mathrm{nm}`$

Same steps:

```math
\begin{aligned}
V_{640}
&= \frac{2\pi (1.5\times 10^{-6})}{640\times 10^{-9}}(0.12) \\
&= 2\pi\left(\frac{1.5}{640}\times 10^{3}\right)(0.12) \\
&\approx 2\pi(2.344)(0.12) \\
&\approx 1.77.
\end{aligned}
```

$`1.77 < 2.405`$ → **single‑mode**.

### 6.4 Practical takeaway for Approach A

- “Single‑mode” is always **wavelength‑dependent**.
- A telecom SMF that is single‑mode at 1550 nm is often multimode at 488/640 nm.

---

## 7) How many modes does an MMF support? (this is why speckle exists)

Once $V\gg 1$, the number of guided modes becomes huge.

### 7.1 Step‑index estimate (large V)

A widely used estimate is:

```math
M_{\mathrm{SI}} \approx \frac{V^2}{2}.
```

Important nuance: different authors count polarization degeneracy differently; factors of 2 are common.
For our use (showing that $M$ is *enormous*), that ambiguity does not matter.

### 7.2 Graded‑index estimate (power‑law profile)

For the power‑law GI model above, one result is:

```math
M_{\mathrm{GI}} \approx \left(\frac{\alpha}{\alpha+2}\right)\frac{V^2}{2}.
```

So for parabolic GI ($`\alpha=2`$):

```math
M_{\mathrm{GI}} \approx \frac{V^2}{4}.
```

### 7.3 Toy numeric example: 400‑µm‑core MMF at 640 nm (repo‑relevant)

Use values consistent with `configs/illumination_mmf_500us.yaml`:

- $`\lambda = 640\thinspace \mathrm{nm}`$
- $`\mathrm{NA}=0.22`$
- core diameter $400\thinspace \mu\mathrm{m}`$ → radius $`a=200\thinspace \mu\mathrm{m}`$

Compute $V$:

```math
\begin{aligned}
V
&= \frac{2\pi a}{\lambda}\mathrm{NA} \\
&= \frac{2\pi (200\times 10^{-6})}{640\times 10^{-9}}(0.22) \\
&= 2\pi\left(\frac{200}{640}\times 10^{3}\right)(0.22) \\
&\approx 2\pi(312.5)(0.22) \\
&\approx 432.
\end{aligned}
```

Then:

```math
M_{\mathrm{SI}} \approx \frac{432^2}{2} \approx 9.3\times 10^4.
```

So: **tens of thousands** of spatial modes.

This is the “MMF superpower” (many degrees of freedom) and the “MMF curse” (lots of interference).

### 7.4 Launch conditions: underfill vs overfill (why “M is huge” is not the whole story)

Even if the fiber *supports* $M$ modes, you might not *excite* them all.

Hecht describes this with “underfilled” vs “overfilled” launch:

- **Underfilled:** input beam occupies a small part of the core and/or a narrow angular cone → mostly low‑order modes.
- **Overfilled:** input beam fills the core and acceptance cone → many modes, including high‑order ones.

For Approach B, this matters because:

- the instantaneous speckle statistics depend on how many modes actually carry power,
- the **time dynamics** of speckle under a scrambler depend on how strongly you excite high‑order modes.

---

## 8) A short Gaussian‑beam detour (because coupling is always about angles)

Chapter 11 of the Optics “f2f” text emphasizes a simple relation for a Gaussian beam:

```math
\Delta\theta \approx \frac{\lambda}{\pi w_0},
```

where:

- $`w_0`$ is the beam waist radius (m),
- $`\Delta\theta`$ is the far‑field divergence half‑angle (radians).

This is useful because fiber acceptance is also an angle constraint:

- the fiber accepts rays up to roughly $`\theta_{\max}\approx \arcsin(\mathrm{NA})`$ in air.

**Toy example (why focusing matters):**

At $`\lambda=640\thinspace \mathrm{nm}`$:

- if $`w_0 = 5\thinspace \mu\mathrm{m}`$,
  ```math
  \Delta\theta \approx \frac{640\times 10^{-9}}{\pi(5\times 10^{-6})}\approx 0.041 \ \mathrm{rad}\approx 2.3^\circ.
  ```
- if $`w_0 = 1\thinspace \mu\mathrm{m}`$,
  ```math
  \Delta\theta \approx 0.20\ \mathrm{rad}\approx 11^\circ.
  ```

So changing the waist by a few µm can move you from “comfortably within NA” to “hitting NA limits”.

---

## 9) Why MMF produces speckle: the one equation to memorize

Write the (complex) output field as a coherent sum of modes:

```math
U(\mathbf{r}) = \sum_{m=1}^{M} a_m\thinspace \psi_m(\mathbf{r})\thinspace e^{i\phi_m}.
```

Interpret every piece:

- $`\mathbf{r}=(x,y)`$ : transverse coordinate at the output facet.
- $`\psi_m(\mathbf{r})`$ : transverse field shape of mode $m$.
- $`a_m \ge 0`$ : amplitude in mode $m$ at the output.
- $`\phi_m`$ : phase accumulated by mode $m$ (depends on fiber bends, launch, temperature, etc.).

### 9.1 Turn field into intensity (and watch “cross terms” appear)

Intensity is magnitude‑squared:

```math
I(\mathbf{r}) = |U(\mathbf{r})|^2 = U(\mathbf{r})\thinspace U^*(\mathbf{r}).
```

Now substitute the sum:

```math
\begin{aligned}
I(\mathbf{r})
&= \left(\sum_m a_m\psi_m e^{i\phi_m}\right)\left(\sum_n a_n\psi_n^* e^{-i\phi_n}\right) \\
&= \sum_m a_m^2|\psi_m|^2 \;+\; \sum_{m\ne n} a_m a_n\thinspace \psi_m\psi_n^*\thinspace e^{i(\phi_m-\phi_n)}.
\end{aligned}
```

Two important parts:

- the $`\sum_m`$ term is a sum of modal **intensities** (always positive),
- the $`\sum_{m\ne n}`$ term contains the **interference** (the speckle‑producing part).

### 9.2 Two‑mode toy example (so the algebra feels real)

Let:

```math
U = a_1e^{i\phi_1}+a_2e^{i\phi_2}.
```

Then:

```math
\begin{aligned}
I &= |U|^2 = U U^* \\
  &= (a_1 e^{i\phi_1} + a_2 e^{i\phi_2})(a_1 e^{-i\phi_1} + a_2 e^{-i\phi_2})\\
  &= a_1^2+a_2^2 + 2a_1a_2\cos(\phi_1-\phi_2).
\end{aligned}
```

If $`a_1=a_2=a`$:

- maximum when $`\cos(\Delta\phi)=+1`$: $`I_{\max}=4a^2`$
- minimum when $`\cos(\Delta\phi)=-1`$: $`I_{\min}=0`$

So even *two* coherent contributions can swing from zero to four‑times intensity.
An MMF has **thousands**.

---

## 10) “Speckle contrast” and why averaging helps

Speckle contrast is typically defined as:

```math
C \equiv \frac{\sigma_I}{\langle I\rangle},
```

where:

- $`\langle I\rangle`$ is mean intensity (spatial or ensemble mean),
- $`\sigma_I`$ is standard deviation of intensity.

A common engineering approximation is:

> If you average $N$ **independent** speckle patterns, contrast scales like $C\sim 1/\sqrt{N}$.

In this repo we split:

```math
N_{\mathrm{eff}} \approx N_t\thinspace N_\lambda\thinspace N_{\mathrm{pol}}\thinspace N_{\mathrm{angle}},
\qquad
C \approx \frac{1}{\sqrt{N_{\mathrm{eff}}}}.
```

Interpret the factors:

- $`N_t`$ : time diversity (scrambler or fast mode coupling during exposure)
- $`N_\lambda`$ : spectral diversity (finite linewidth; multiple wavelengths)
- $`N_{\mathrm{pol}}`$ : polarization diversity (often up to 2)
- $`N_{\mathrm{angle}}`$ : angle diversity (scan pupil / change launch angle)

The rest of this document is mostly about estimating $`N_\lambda`$ and understanding why it depends on
fiber type (SI vs GI), length, and NA.

---

## 11) Temporal coherence: linewidth → coherence length (define every step)

### 11.1 Coherence length from linewidth (order‑of‑magnitude)

A widely used estimate is:

```math
L_c \sim \frac{\lambda_0^2}{\Delta\lambda}.
```

Interpretation:

- wider spectrum ($`\Delta\lambda\uparrow`$) → shorter coherence length ($`L_c\downarrow`$).

In a medium of index $n$, a “coherence length in the medium” is shorter by about $n$:

```math
L_{c,\mathrm{med}} \sim \frac{\lambda_0^2}{n\thinspace \Delta\lambda}.
```

### 11.2 Toy numeric example at 640 nm

Take $`\lambda_0 = 640\thinspace \mathrm{nm}`$ so $`\lambda_0^2 = (640\thinspace \mathrm{nm})^2 = 409600\thinspace \mathrm{nm}^2`$.

Now divide by linewidth:

- $`\Delta\lambda = 1\thinspace \mathrm{nm}`$  
  $`L_c \sim 409600\thinspace \mathrm{nm} \approx 0.41\thinspace \mathrm{mm}`$

- $`\Delta\lambda = 2\thinspace \mathrm{nm}`$  
  $`L_c \sim 0.20\thinspace \mathrm{mm}`$

- $`\Delta\lambda = 10\thinspace \mathrm{nm}`$  
  $`L_c \sim 0.041\thinspace \mathrm{mm} = 41\thinspace \mu\mathrm{m}`$

So “nm‑scale linewidth” means **sub‑millimeter coherence length**.

### 11.3 Illustration: coherence time vs intermodal delay (key to MMF speckle)

![Coherence envelope vs modal delay](figures/coherence_vs_delay.svg)

Interpretation:

- If two modal contributions differ in optical path by $`\Delta\mathrm{OPL}`$,
  the corresponding time delay is $`\Delta\tau \approx \Delta\mathrm{OPL}/c`$.
- If $`\Delta\mathrm{OPL} \gg L_c`$ (equivalently $`\Delta\tau \gg \tau_c`$),
  then the interference cross terms average out when you integrate over the spectrum.

This is exactly why Agrawal notes that “modal noise” in MMF links disappears for broad sources:
intermodal interference requires coherence time longer than intermodal delay.

---

## 12) Step‑index upper bound on MMF optical‑path spread (derive it in slow motion)

We want an estimate for $`\Delta\mathrm{OPL}`$ across guided paths, because it controls spectral averaging.

Model (meridional rays, step‑index):

- Fiber physical length is $L$.
- A ray inside the core makes an angle $`\theta`$ to the axis.
- Geometric path length is $`L/\cos\theta`$.
- Optical path length multiplies by $`n_1`$.

So:

```math
\mathrm{OPL}(\theta) = n_1\thinspace \frac{L}{\cos\theta}.
```

The axial ray has $`\theta=0`$:

```math
\mathrm{OPL}(0) = n_1 L.
```

So the excess is:

```math
\Delta\mathrm{OPL}(\theta) = n_1 L\left(\frac{1}{\cos\theta}-1\right).
```

### 12.1 Small‑angle approximation (show the exact algebra move)

For small $`\theta`$ (radians):

```math
\cos\theta \approx 1-\frac{\theta^2}{2}.
```

We also need $`1/\cos\theta`$. Use the approximation:

```math
\frac{1}{1-x} \approx 1+x \qquad \text{when } |x|\ll 1.
```

Here $`x=\theta^2/2`$, so:

```math
\frac{1}{\cos\theta} \approx 1+\frac{\theta^2}{2}.
```

Insert into $`\Delta\mathrm{OPL}`$:

```math
\Delta\mathrm{OPL}(\theta) \approx n_1 L\left(\frac{\theta^2}{2}\right).
```

### 12.2 What is the maximum internal angle?

Externally, the fiber accepts up to $`\theta_{\max}`$ with $`\sin\theta_{\max}\approx\mathrm{NA}`$ (air).

Inside the core, Snell’s law gives approximately:

```math
\sin\theta_{\max,\mathrm{core}} \approx \frac{\mathrm{NA}}{n_1}.
```

For small angles, $`\sin\theta\approx\theta`$, so:

```math
\theta_{\max,\mathrm{core}} \approx \frac{\mathrm{NA}}{n_1}.
```

### 12.3 Final step: plug the maximum angle into the formula

Use $`\theta=\theta_{\max,\mathrm{core}}`$:

```math
\Delta\mathrm{OPL}
\approx n_1 L\left(\frac{1}{2}\frac{\mathrm{NA}^2}{n_1^2}\right)
= \left(\frac{\mathrm{NA}^2}{2n_1}\right)L.
```

This is the exact rule‑of‑thumb written in `configs/illumination_mmf_500us.yaml`.

### 12.4 Toy numeric example (repo‑typical values)

Use:

- $`\mathrm{NA}=0.22`$
- $`n_1=1.46`$
- $`L=3\thinspace \mathrm{m}`$

Compute step by step:

1) square NA: $`0.22^2 = 0.0484`$

2) divide by $`2n_1`$:  
   $`2n_1 = 2.92`$  
   $`0.0484/2.92 \approx 0.0166`$

3) multiply by L:  
   $`\Delta\mathrm{OPL} \approx 0.0166\times 3 \approx 0.050\thinspace \mathrm{m}`$

So $`\Delta\mathrm{OPL}\approx 5\thinspace \mathrm{cm}`$.

Convert to delay:

```math
\Delta\tau \approx \frac{\Delta\mathrm{OPL}}{c}
\approx \frac{0.05}{3\times 10^8}
\approx 1.7\times 10^{-10}\thinspace \mathrm{s}
= 170\thinspace \mathrm{ps}.
```

---

## 13) Spectral decorrelation width: the bridge from “fiber delay” to “speckle averaging”

The key idea:

- two wavelengths separated by $`\Delta\lambda`$ have different wavenumbers $k$,
- that changes the interference phases across paths differing by $`\Delta\mathrm{OPL}`$,
- once those phases change by about $2\pi$, the speckle pattern decorrelates.

### 13.1 Convert wavelength change to wavenumber change (show the derivative)

Wavenumber (vacuum):

```math
k(\lambda)=\frac{2\pi}{\lambda}.
```

Differentiate:

```math
\frac{dk}{d\lambda} = -\frac{2\pi}{\lambda^2}.
```

For a small change $`\Delta\lambda`$ around $`\lambda_0`$:

```math
\Delta k \approx \left|\frac{dk}{d\lambda}\right|_{\lambda_0}\Delta\lambda
= \frac{2\pi}{\lambda_0^2}\Delta\lambda.
```

### 13.2 Set the “decorrelation condition”

A common criterion is:

```math
\Delta k \thinspace \Delta\mathrm{OPL} \sim 2\pi.
```

Insert $`\Delta k`$:

```math
\frac{2\pi}{\lambda_0^2}\Delta\lambda_c \thinspace \Delta\mathrm{OPL} \sim 2\pi.
```

Cancel $`2\pi`$ on both sides:

```math
\Delta\lambda_c \sim \frac{\lambda_0^2}{\Delta\mathrm{OPL}}.
```

That’s the key formula used in the repo’s MMF linewidth estimates.

### 13.3 Toy numeric example (same fiber as Section 12)

We found $`\Delta\mathrm{OPL}\approx 0.05\thinspace \mathrm{m}`$.

Use $`\lambda_0=640\thinspace \mathrm{nm}`$:

1) square: $`\lambda_0^2 = 409600\thinspace \mathrm{nm}^2`$

2) divide by $`\Delta\mathrm{OPL}`$ (convert $0.05\thinspace \mathrm{m}`$ to nm):  
   $`0.05\thinspace \mathrm{m} = 5\times 10^{7}\thinspace \mathrm{nm}`$

3) compute:
   ```math
   \Delta\lambda_c \sim \frac{4.096\times 10^{5}}{5\times 10^{7}}\ \mathrm{nm}
   \approx 8.2\times 10^{-3}\ \mathrm{nm}.
   ```

So $`\Delta\lambda_c`$ is about $`0.008\thinspace \mathrm{nm}`$.

### 13.4 Convert decorrelation width to “number of independent spectral looks”

If your source has effective spectral span $`\Delta\lambda_{\mathrm{src}}`$, then:

```math
N_\lambda \sim \frac{\Delta\lambda_{\mathrm{src}}}{\Delta\lambda_c}.
```

Toy examples:

- If $`\Delta\lambda_{\mathrm{src}} = 1\thinspace \mathrm{nm}`$ and $`\Delta\lambda_c=0.008\thinspace \mathrm{nm}`$  
  $`N_\lambda \sim 125`$

- If $`\Delta\lambda_{\mathrm{src}} = 2\thinspace \mathrm{nm}`$  
  $`N_\lambda \sim 250`$

This is why linewidth can be such a powerful knob for Approach B.

### 13.5 The graded‑index “gotcha” (why your procurement questions matter)

GI fibers reduce intermodal delay. Model that as:

```math
\Delta\mathrm{OPL}_{\mathrm{GI}} = s\thinspace \Delta\mathrm{OPL}_{\mathrm{SI}},
\qquad 0<s<1.
```

Then:

```math
\Delta\lambda_{c,\mathrm{GI}} \sim \frac{\lambda_0^2}{s\thinspace \Delta\mathrm{OPL}_{\mathrm{SI}}}
= \frac{1}{s}\Delta\lambda_{c,\mathrm{SI}}.
```

So if GI reduces delay by 10× ($s=0.1$), $`\Delta\lambda_c`$ becomes 10× larger and **$`N_\lambda`$ becomes 10× smaller**.

That is why a fiber sold as a “homogenizer” might be GI:
it is good at spatial mixing, but it may reduce the “free” spectral averaging you were counting on.

(Your code models this explicitly via `modal_delay_scale` in `src/illumination/mmf_speckle.py`.)

---

## 14) Time averaging: why a scrambler alone often cannot save 500 µs

If a scrambler creates new (approximately independent) speckle realizations at rate $`f_{\mathrm{scr}}`$,
then during an exposure of duration $`\tau`$:

```math
N_t \approx f_{\mathrm{scr}}\tau.
```

Toy example:

- $`f_{\mathrm{scr}}=10\thinspace \mathrm{kHz}`$
- $`\tau=500\thinspace \mu\mathrm{s}=5\times 10^{-4}\thinspace \mathrm{s}`$

Then:

```math
N_t \approx 10^{4}\times 5\times 10^{-4} \approx 5.
```

So **time averaging alone** gives only about a factor $`\sqrt{5}\approx 2.2`$ improvement in contrast.
That’s why Approach B leans heavily on **spectral diversity** (and possibly polarization/angle).

---

## 15) A practical procurement/engineering checklist (SMF vs MMF)

### 15.1 If you pursue Approach A (SMF per wavelength)

Ask / verify:

- Fiber is **single‑mode at your wavelength** (check $V<2.405$ using vendor’s MFD/core/NA spec).
- Mode field diameter (MFD) at 488 and 640 nm (affects collimator choice and coupling).

### 15.2 If you pursue Approach B (MMF)

Ask / verify:

1) **Step‑index or graded‑index?**  
   If the vendor says “GI” or “parabolic”, treat it as a warning that $`N_\lambda`$ may be smaller.

2) Core diameter and NA (sets $V$, $M$, and acceptance).

3) Length (sets $`\Delta\mathrm{OPL}`$ linearly).

4) Any integrated mode‑mixing / diffusion features (good for spatial stability, may affect temporal dynamics).

5) Whether the “specified NA” is for the core or includes cladding guidance (some specialty fibers are tricky).

### 15.3 If your goal is: “smooth illumination within 500 µs”

Translate into a target:

- Choose a contrast target (example: $`C\lesssim 0.1`$).
- That implies $`N_{\mathrm{eff}}\gtrsim 100`$.

Then check feasibility:

- $`N_t`$ from your scrambler during 500 µs (often small).
- $`N_\lambda`$ from your linewidth and fiber delay spread (can be large).
- $`N_{\mathrm{pol}}`$ (up to 2 if polarization is randomized).
- $`N_{\mathrm{angle}}`$ (if you scan/rotate pupil).

---

## 16) Where this connects into the repo (so the primer becomes actionable)

- `configs/cni_laser_inquiry.yaml` encodes the **two approaches** and constraints.
- `configs/illumination_mmf_500us.yaml` encodes the MMF “500 µs” design sweep assumptions.
- `src/illumination/mmf_speckle.py` implements the same order‑of‑magnitude relations:
  - $V$ number,
  - $M\sim V^2/2$,
  - $`\Delta\tau \sim \mathrm{NA}^2L/(2n_1c)`$,
  - $`\Delta\lambda_c \sim \lambda^2/\Delta\mathrm{OPL}`$,
  - $`N_\lambda \sim \Delta\lambda_{\mathrm{src}}/\Delta\lambda_c`$.

---

## 17) Limitations (what this primer is *not* doing)

This is deliberately “engineering‑level”:

- It does **not** derive the full LP mode solutions (Bessel functions, dispersion equations).
- It does **not** model wavelength‑dependent material dispersion in detail.
- It treats many results as order‑of‑magnitude (good enough to size $`N_\lambda`$, choose fiber type, and sanity‑check vendor claims).

If you ever need higher accuracy for $`\Delta\mathrm{OPL}`$ in a specific GI fiber,
you will want either:
- vendor dispersion specs, or
- an empirical measurement (e.g., speckle spectral correlation measurement).

---

## 18) Sources used (project files)

This primer is based primarily on:

- G.P. Agrawal, *Fiber‑Optic Communication Systems* (2021 PDF in this repo).
- Dan Oron course notes, “guided waves” chapter PDF in this repo.
- E. Hecht, *Optics* (5th ed.) section on fiber optics and V‑number.
- Chapter 11 “Light propagation: beams and guides” from the Optics f2f book (PDF chapter).
- J.W. Goodman, *Speckle Phenomena in Optics* (for GI profile intuition and modal speckle context).

(See the project root for the corresponding PDF/HTML files.)
