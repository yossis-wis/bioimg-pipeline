# Guided-wave primer for this repo: single-mode vs multimode fiber (SMF vs MMF)

> **Scope (why you are reading this):**
> This is a **chapter-like, equation-first** explanation of *guided waves in fibers* aimed at the two
> illumination-delivery approaches used throughout this repo:
>
> - **Approach A:** separate **single-mode fiber** outputs per wavelength, combined in free space.
> - **Approach B:** **wide-linewidth** free-space lasers combined into a **single multimode fiber** (MMF)
>   + scrambler for speckle averaging.
>
> **Sources (provided in this project):**
> - Agrawal, *Fiber-Optic Communication Systems* (5th ed., 2021) — see `Fiber‐Optic Communication Systems - 2021 - Agrawal.pdf` in this project’s files.
> - Dan Oron course notes: *Guided wave optics* — see `guided waves.pdf` in this project’s files.
>
> **Printing note:** designed to print cleanly on a **black-and-white laser printer**.
> All diagrams are ASCII; no dependence on color.

---

## 0) What you should get out of this

After reading, you should be able to do the following *without hand-waving*:

1. Given a fiber core radius $a$, wavelength $\lambda$, and numerical aperture $\mathrm{NA}$, compute the
   **V-number** $V$ and decide if the fiber is **single-mode** or **multimode**.
2. For a multimode fiber, estimate how many modes exist (order of magnitude), and understand why that
   immediately implies **speckle** unless something makes the modes add **incoherently**.
3. Given an MMF length $L$, $\mathrm{NA}$ and core index $`n_1`$, estimate an **optical-path-length spread**
   $`\Delta\mathrm{OPL}`$ and from it estimate:
   - the **spectral decorrelation width** $`\Delta\lambda_c`$,
   - the number of independent spectral “looks” $`N_\lambda`$ inside one exposure.
4. Combine time/spectral/polarization/angle diversity into an **effective** number of averages
   $`N_{\mathrm{eff}}`$ and estimate speckle contrast $`C \sim 1/\sqrt{N_{\mathrm{eff}}}`$.

Throughout, we use representative values from this repo’s configs, especially:

- `configs/cni_laser_inquiry.yaml` (defines Approach A vs B concepts)
- `configs/illumination_mmf_500us.yaml` (defines 500 µs MMF design sweep assumptions)

---

## 1) The two approaches in this repo (in one picture)

The repo is exploring two **delivery** concepts (not “two lasers”):

### Approach A: single-mode fiber per wavelength (stable spatial mode)

```
[Laser 640]--(SMF)->(collimator)--\
                                 +--> [combine in free space] --> [relay to objective]
[Laser 488]--(SMF)->(collimator)--/
```

Key feature: each wavelength emerges in essentially one spatial mode (a smooth, near-Gaussian field).

### Approach B: wide-linewidth lasers into one multimode fiber (many modes + averaging)

```
[Laser 640 free-space]--\
                         +--> [combine] --> [MMF] --> [scrambler] --> [collimate + stop] --> [objective]
[Laser 488 free-space]--/
```

Key feature: the MMF supports **many guided modes**, which (if coherent) interfere at the output to form speckle.
The whole point of the wide linewidth + scrambler is to make the observed intensity behave more like an
**incoherent sum/average** of many independent patterns, which reduces speckle contrast during short exposures.

---

## 2) Step-index waveguide geometry and the three core parameters

Both Agrawal and the Oron notes use the same physical model for an ordinary step-index fiber:

- **core index:** $`n_1`$
- **cladding index:** $`n_2`$ with $`n_1 > n_2`$
- **core radius:** $a$

A “step-index” profile means the refractive index jumps at $r=a$:

```math
n(r) =
\begin{cases}
 n_1, & r < a \\
 n_2, & r > a.
\end{cases}
```

Everything that matters for “single-mode vs multimode” at a given wavelength can be boiled down to:

1. the **index contrast** (often written as $\Delta$),
2. the **core radius** $a$, and
3. the **vacuum wavelength** $\lambda$.

---

## 3) Numerical aperture (NA): ray-optics meaning and wave-optics meaning

### 3.1 Ray-optics definition

The **numerical aperture** $\mathrm{NA}$ is (by definition) the sine of the maximum acceptance half-angle in air:

```math
\mathrm{NA} \equiv n_0\sin\theta_{\max}.
```

For air, $`n_0\approx 1`$, so $`\mathrm{NA} \approx \sin\theta_{\max}`$.

For a step-index fiber, total internal reflection at the core–cladding interface leads to the well-known result:

```math
\mathrm{NA} = \sqrt{n_1^2 - n_2^2}.
```

**What this means operationally:** if you inject a free-space beam whose half-angle cone is smaller than
$`\theta_{\max}`$, the fiber can accept it (in the geometric-optics sense).

### 3.2 Index-contrast parameter $\Delta$ (common in fiber theory)

A very common dimensionless parameter is the (approximate) fractional index contrast $\Delta$.
One typical definition is:

```math
\Delta \equiv \frac{n_1^2 - n_2^2}{2n_1^2}.
```

If $\Delta \ll 1$ (true for most silica fibers), then you can show:

```math
\mathrm{NA} \approx n_1\sqrt{2\Delta}.
```

#### Toy numeric example: compute NA from indices

Take a “visible-ish SMF” index pair such as:

- $`n_1 = 1.460`$
- $`n_2 = 1.455`$

Compute:

1) squares: $`n_1^2 = 1.460^2 \approx 2.1316`$, $`n_2^2 = 1.455^2 \approx 2.1170`$.

2) subtract: $`n_1^2-n_2^2 \approx 0.0146`$.

3) sqrt: $\mathrm{NA} \approx \sqrt{0.0146} \approx 0.121$.

So the acceptance half-angle in air is about:

```math
\theta_{\max} \approx \arcsin(0.121) \approx 6.9^\circ.
```

This “small angle” fact is why small-angle approximations show up everywhere.

---

## 4) Wave-optics view: what a “mode” is

Ray optics tells you whether light is trapped, but it does *not* tell you the full field.
Wave optics tells you that inside a waveguide, the electromagnetic field can propagate in special patterns
that reproduce after any distance $z$ up to a phase factor.

A standard separation-of-variables form for a single-frequency field is:

```math
\mathbf{E}(x,y,z,t) = \Re\Big\{\mathbf{e}(x,y)\,\exp\big(i\beta z - i\omega t\big)\Big\}.
```

Every symbol here matters:

- $\mathbf{E}$ is the **real** electric field.
- $\mathbf{e}(x,y)$ is the **transverse mode profile** (complex in general).
- $\omega = 2\pi f$ is the angular frequency.
- $\beta$ is the **propagation constant** along $z$ (units: rad/m).
  It plays the role of “axial wavenumber.”

A **guided mode** is one of the discrete solutions $\mathbf{e}(x,y)$ that satisfies Maxwell + boundary conditions.
Each allowed mode comes with its own $`\beta_m`$.

### Why “discrete” modes appear (the slab-waveguide analogy)

The Oron notes develop this first for a slab waveguide: a plane wave bounces between the interfaces.
A mode exists only if the round-trip phase is self-consistent (phase matched), which forces **quantized** angles.
In full wave optics, that quantization becomes quantized $\beta$.

You do not need the full derivation for this project, but you *do* need the consequence:

> **A waveguide supports only certain allowed transverse field patterns.**
> Those patterns are the “modes,” and “single-mode” literally means only the lowest-order one exists.

---

## 5) The V-number: one dimensionless number that predicts single-mode vs multimode

### 5.1 Definition

For a step-index **cylindrical** fiber, define the **vacuum** wavenumber:

```math
k_0 \equiv \frac{2\pi}{\lambda}.
```

Then define the **normalized frequency** (V-number):

```math
V \equiv k_0 a\,\sqrt{n_1^2-n_2^2} = \frac{2\pi a}{\lambda}\,\mathrm{NA}.
```

Interpretation:

- Bigger core ($a\uparrow$) → more room transversely → more modes.
- Shorter wavelength ($\lambda\downarrow$) → finer transverse structure possible → more modes.
- Larger NA (bigger index step) → tighter confinement possible → more modes.

### 5.2 Single-mode cutoff (step-index fiber)

A classic result (Agrawal + any fiber text) is:

> A step-index fiber is strictly single-mode when $V < 2.405$.

The number $2.405$ is the first zero of the Bessel function $`J_0`$.
It appears because the exact solution in cylindrical coordinates uses Bessel functions in the core and
modified Bessel functions in the cladding.

### 5.3 Slab-waveguide cutoff (why Oron’s notes mention $\pi/2$)

For a **slab** waveguide (1D confinement), the cutoff for the first higher-order mode occurs at a different number.
In the Oron notes you see a condition equivalent to:

```math
V_{\mathrm{slab}} < \frac{\pi}{2}
```

for single-mode operation.

**Do not mix these constants.** They refer to different geometries:

- slab: one transverse dimension confined → cutoff $\sim \pi/2$.
- fiber: two transverse dimensions confined → cutoff $\sim 2.405$.

The take-home is the same: there is a geometry-dependent “order-unity” cutoff value.

---

## 6) Toy calculations: “is my fiber single-mode at 488 and 640?”

This is **the** practical question behind Approach A.

### 6.1 Example A: a “visible SMF” that stays single-mode at 488 and 640

Assume:

- core radius $a = 1.5\thinspace\mu\mathrm{m}$
- $\mathrm{NA} = 0.12$

Compute $V$ at two wavelengths.

#### Step 1: convert units

- $1\thinspace\mu\mathrm{m} = 10^{-6}\thinspace\mathrm{m}$
- $1\thinspace\mathrm{nm} = 10^{-9}\thinspace\mathrm{m}$

So:

- $a = 1.5\times 10^{-6}\thinspace\mathrm{m}$
- $`\lambda_{488} = 488\times 10^{-9}\thinspace\mathrm{m}`$
- $`\lambda_{640} = 640\times 10^{-9}\thinspace\mathrm{m}`$

#### Step 2: apply $V = (2\pi a/\lambda)\,\mathrm{NA}$

For $488\thinspace\mathrm{nm}$:

```math
\begin{aligned}
V_{488}
&= \frac{2\pi\,(1.5\times 10^{-6})}{488\times 10^{-9}}\,(0.12)\\
&= 2\pi\,\Big(\frac{1.5}{488}\times 10^{3}\Big)\,(0.12)\\
&\approx 2\pi\,(3.074)\,(0.12)\\
&\approx 2.32.
\end{aligned}
```

For $640\thinspace\mathrm{nm}$:

```math
\begin{aligned}
V_{640}
&= \frac{2\pi\,(1.5\times 10^{-6})}{640\times 10^{-9}}\,(0.12)\\
&= 2\pi\,\Big(\frac{1.5}{640}\times 10^{3}\Big)\,(0.12)\\
&\approx 2\pi\,(2.344)\,(0.12)\\
&\approx 1.77.
\end{aligned}
```

#### Step 3: compare to the single-mode cutoff 2.405

- $`V_{488} \approx 2.32 < 2.405`$ → **single-mode** (barely, but yes).
- $`V_{640} \approx 1.77 < 2.405`$ → **single-mode**.

This is why visible SMF cores are *small*: short wavelengths drive $V$ up.

### 6.2 Example B: telecom “single-mode” fiber becomes multimode in the visible

A telecom SMF might have roughly:

- $a \approx 4.1\thinspace\mu\mathrm{m}$
- $\mathrm{NA} \approx 0.14$

At $\lambda=1550\thinspace\mathrm{nm}$ you get $V\sim 2.3$ (single-mode).
But at $640\thinspace\mathrm{nm}$:

```math
V_{640} \approx \frac{2\pi (4.1\times 10^{-6})}{640\times 10^{-9}}\,(0.14) \approx 5.6,
```

which is strongly **multimode**.

**Practical consequence for Approach A:** you generally cannot use one off-the-shelf telecom SMF patch cable
for both $488\thinspace\mathrm{nm}$ and $640\thinspace\mathrm{nm}$ and still stay single-mode.

---

## 7) How many modes does an MMF support?

Once $V\gg 1$, the fiber supports a large number of guided modes.
A widely used step-index estimate is:

```math
M \approx \frac{V^2}{2}
```

where $M$ is the number of guided modes (order-of-magnitude; counting conventions differ by factors near 2).
For a graded-index MMF, an often-quoted estimate is smaller by about a factor of two:

```math
M_{\mathrm{GI}} \sim \frac{V^2}{4}.
```

### Toy numeric example: 400 µm-core MMF at 640 nm

Use values consistent with `configs/illumination_mmf_500us.yaml`:

- $\lambda = 640\thinspace\mathrm{nm}$
- $\mathrm{NA} = 0.22$

Suppose you consider a large-core MMF with diameter $400\thinspace\mu\mathrm{m}$, i.e. radius
$a = 200\thinspace\mu\mathrm{m}$.

Compute $V$:

```math
\begin{aligned}
V
&= \frac{2\pi a}{\lambda}\,\mathrm{NA}\\
&= \frac{2\pi (200\times 10^{-6})}{640\times 10^{-9}}\,(0.22)\\
&= 2\pi\,\Big(\frac{200}{640}\times 10^{3}\Big)\,(0.22)\\
&\approx 2\pi\,(312.5)\,(0.22)\\
&\approx 432.
\end{aligned}
```

Then

```math
M \approx \frac{V^2}{2} \approx \frac{(432)^2}{2} \approx 9.3\times 10^4.
```

So: **tens of thousands** of spatial modes are supported.

This is the “MMF superpower” (many degrees of freedom), and also the “MMF curse” (lots of interference speckle).

---

## 8) Why multimode fiber produces speckle (the key equation)

At the MMF output plane, the complex field can be written as a coherent sum over guided modes:

```math
U(\mathbf{r}) = \sum_{m=1}^{M} a_m\,\psi_m(\mathbf{r})\,\exp(i\phi_m).
```

Interpret each factor:

- $U(\mathbf{r})$ : complex field at transverse coordinate $\mathbf{r}=(x,y)$.
- $`\psi_m(\mathbf{r})`$ : normalized transverse mode shape.
- $`a_m`$ : (real, nonnegative) modal amplitude at the output.
- $`\phi_m`$ : modal phase (depends on launch conditions, path length, bending, temperature, etc.).

The measured intensity is:

```math
I(\mathbf{r}) = |U(\mathbf{r})|^2.
```

Expanding $|\cdot|^2$ exposes the interference explicitly:

```math
I(\mathbf{r}) = \sum_m a_m^2|\psi_m(\mathbf{r})|^2
+ \sum_{m\neq n} a_m a_n\,\psi_m(\mathbf{r})\psi_n^*(\mathbf{r})\,\exp\big(i(\phi_m-\phi_n)\big).
```

- The first term is a sum of modal **intensities**.
- The second term is the sum of **cross terms** (interference).

When many cross terms with effectively random phase differences contribute, $I(\mathbf{r})$ becomes a
high-contrast granular pattern: speckle.

### 8.1 Two-mode toy example (so the algebra is concrete)

Let the field be just two phasors:

```math
U = a_1 e^{i\phi_1} + a_2 e^{i\phi_2}.
```

Compute intensity:

```math
\begin{aligned}
I &= |U|^2 = U U^* \\
  &= (a_1 e^{i\phi_1} + a_2 e^{i\phi_2})(a_1 e^{-i\phi_1} + a_2 e^{-i\phi_2})\\
  &= a_1^2 + a_2^2 + a_1 a_2\,\big(e^{i(\phi_1-\phi_2)} + e^{-i(\phi_1-\phi_2)}\big)\\
  &= a_1^2 + a_2^2 + 2 a_1 a_2\cos(\Delta\phi),
\end{aligned}
```

where $`\Delta\phi \equiv \phi_1-\phi_2`$.

If $`a_1=a_2=a`$:

- max intensity when $\cos(\Delta\phi)=+1$: $`I_{\max}=4a^2`$
- min intensity when $\cos(\Delta\phi)=-1$: $`I_{\min}=0`$

So just two coherent contributions can already produce huge fluctuations.
An MMF has thousands of them.

---

## 9) Speckle contrast and “effective number of independent looks”

A standard engineering approximation (used throughout this repo) is:

> If you average $N$ **independent** speckle patterns (incoherent sum or time average),
> speckle contrast scales like $C \sim 1/\sqrt{N}$.

In this repo we factor $N$ into multiplicative diversity mechanisms:

```math
N_{\mathrm{eff}} \approx N_t\,N_\lambda\,N_{\mathrm{pol}}\,N_{\mathrm{angle}}.
```

Where:

- $`N_t`$ : number of independent patterns during the exposure (time diversity).
- $`N_\lambda`$ : number of independent spectral “looks” (spectral diversity).
- $`N_{\mathrm{pol}}`$ : polarization diversity (often up to 2).
- $`N_{\mathrm{angle}}`$ : angle diversity (e.g. scanning pupil angle).

Then a first-pass contrast estimate is:

```math
C \approx \frac{1}{\sqrt{N_{\mathrm{eff}}}}.
```

This is the conceptual bridge between **fiber mode theory** and your practical question:

> “Will MMF illumination be smooth enough during a $500\thinspace\mu\mathrm{s}$ exposure?”

---

## 10) Temporal coherence: why linewidth matters for MMF speckle

### 10.1 From linewidth $\Delta\lambda$ to coherence length $`L_c`$

A useful order-of-magnitude estimate is:

```math
L_c \sim \frac{\lambda_0^2}{\Delta\lambda}.
```

This is a vacuum coherence length. In a medium with refractive index $n$, an effective in-medium coherence
length is reduced by about $n$:

```math
L_{c,\mathrm{med}} \sim \frac{\lambda_0^2}{n\,\Delta\lambda}.
```

**What this means physically:** if two contributions to the field arrive with an optical path difference
much larger than $`L_c`$, they do not maintain a stable phase relationship and their interference washes out.

### 10.2 Toy numeric example at 640 nm

Take $`\lambda_0 = 640\thinspace\mathrm{nm}`$.

Compute $`L_c`$ for a few linewidths:

- $\Delta\lambda = 0.001\thinspace\mathrm{nm}$ → $`L_c \sim 0.41\thinspace\mathrm{m}`$ (very coherent)
- $\Delta\lambda = 0.01\thinspace\mathrm{nm}$ → $`L_c \sim 4.1\thinspace\mathrm{cm}`$
- $\Delta\lambda = 1\thinspace\mathrm{nm}$ → $`L_c \sim 0.41\thinspace\mathrm{mm}`$
- $\Delta\lambda = 2\thinspace\mathrm{nm}$ → $`L_c \sim 0.20\thinspace\mathrm{mm}`$
- $\Delta\lambda = 10\thinspace\mathrm{nm}$ → $`L_c \sim 41\thinspace\mu\mathrm{m}`$

So “wide linewidth” (nm-scale) implies **sub-mm** coherence length.

---

## 11) Optical-path-length spread in an MMF (the key bridge from waveguide to speckle)

Your MMF output speckle depends strongly on whether different guided contributions are mutually coherent.
A convenient way to quantify this is the optical path spread $`\Delta\mathrm{OPL}`$.

### 11.1 Derivation of a step-index upper bound (matches the repo config comment)

Consider a ray propagating in the core at angle $\theta$ relative to the fiber axis.

- The fiber has physical (axial) length $L$.
- The geometric path length of the angled ray is $L/\cos\theta$.
- Optical path length multiplies by refractive index $`n_1`$:

```math
\mathrm{OPL}(\theta) = n_1\,\frac{L}{\cos\theta}.
```

The axial ray has $\theta=0$:

```math
\mathrm{OPL}(0) = n_1 L.
```

So the optical path *excess* is:

```math
\Delta\mathrm{OPL}(\theta) = n_1 L\left(\frac{1}{\cos\theta} - 1\right).
```

Now use a small-angle approximation.
For small $\theta$ (in radians):

```math
\cos\theta \approx 1 - \frac{\theta^2}{2}
\quad\Rightarrow\quad
\frac{1}{\cos\theta} \approx 1 + \frac{\theta^2}{2}.
```

Insert that:

```math
\Delta\mathrm{OPL}(\theta) \approx n_1 L\left(\frac{\theta^2}{2}\right).
```

What is $`\theta_{\max}`$ inside the core?

From Snell’s law at the input, the external acceptance is set by $\mathrm{NA}$.
Inside the core, the maximum internal angle satisfies approximately:

```math
\sin\theta_{\max,\mathrm{core}} \approx \frac{\mathrm{NA}}{n_1}.
```

For small angles, $\sin\theta\approx\theta$, so:

```math
\theta_{\max,\mathrm{core}} \approx \frac{\mathrm{NA}}{n_1}.
```

Therefore, an upper-bound path spread across accepted rays is:

```math
\Delta\mathrm{OPL} \approx n_1 L\left(\frac{1}{2}\frac{\mathrm{NA}^2}{n_1^2}\right)
= \left(\frac{\mathrm{NA}^2}{2n_1}\right) L.
```

This is exactly the rule-of-thumb written in `configs/illumination_mmf_500us.yaml`.

### 11.2 Toy numeric example (again matching repo values)

Use:

- $\mathrm{NA}=0.22$
- $`n_1\approx 1.46`$ (silica core)
- $L=3\thinspace\mathrm{m}$

Compute:

```math
\Delta\mathrm{OPL} \approx \left(\frac{0.22^2}{2\cdot 1.46}\right)\,3
\approx (0.0166)\,3
\approx 0.050\thinspace\mathrm{m}.
```

So $`\Delta\mathrm{OPL}`$ is about $5\thinspace\mathrm{cm}$.

Convert to an equivalent relative delay:

```math
\Delta\tau \approx \frac{\Delta\mathrm{OPL}}{c} \approx \frac{0.05}{3\times 10^8}\approx 1.7\times 10^{-10}\thinspace\mathrm{s} = 170\thinspace\mathrm{ps}.
```

This is a *huge* delay compared to the coherence time of a nm-linewidth source (ps-scale), which is why
wide linewidth can kill intermode interference.

### 11.3 Graded-index nuance

Graded-index MMF is engineered to reduce intermodal delay (good for communications).
For illumination speckle averaging, that means $`\Delta\mathrm{OPL}`$ can be **smaller**, which makes
spectral decorrelation weaker (smaller $`N_\lambda`$ for a given linewidth).

This is not “good” or “bad” universally—it is just a trade:

- GI fiber: better pulse fidelity / less modal dispersion
- SI fiber: larger path spread / easier spectral decorrelation

---

## 12) Spectral decorrelation width $`\Delta\lambda_c`$ (the linewidth→speckle bridge)

A standard estimate used in this repo is:

```math
\Delta\lambda_c \sim \frac{\lambda^2}{\Delta\mathrm{OPL}}.
```

Here is the derivation in slow motion.

### 12.1 Derivation from phase sensitivity

Interference depends on relative phase.
For an optical path difference $`\Delta\mathrm{OPL}`$, the phase difference at wavelength $\lambda$ is:

```math
\Delta\phi(\lambda) = 2\pi\,\frac{\Delta\mathrm{OPL}}{\lambda}.
```

Now change wavelength by a small amount $\delta\lambda$.
Use the first-order approximation:

```math
\frac{1}{\lambda+\delta\lambda} \approx \frac{1}{\lambda} - \frac{\delta\lambda}{\lambda^2}.
```

So the phase change is approximately:

```math
\delta(\Delta\phi) \approx 2\pi\,\Delta\mathrm{OPL}\left(\frac{\delta\lambda}{\lambda^2}\right).
```

Speckle becomes effectively uncorrelated when this phase change is order $2\pi$.
Set $\delta(\Delta\phi)\approx 2\pi$ and cancel $2\pi$:

```math
\Delta\mathrm{OPL}\left(\frac{\delta\lambda}{\lambda^2}\right) \approx 1
\quad\Rightarrow\quad
\delta\lambda \approx \frac{\lambda^2}{\Delta\mathrm{OPL}}.
```

That $\delta\lambda$ is the decorrelation width $`\Delta\lambda_c`$.

### 12.2 Toy numeric example (640 nm, 5 cm spread)

Use $\lambda=640\thinspace\mathrm{nm}$ and $`\Delta\mathrm{OPL}`=5\thinspace\mathrm{cm}$.

Convert $\lambda$ to meters: $640\times 10^{-9}\thinspace\mathrm{m}$.

Compute:

```math
\Delta\lambda_c \approx \frac{(640\times 10^{-9})^2}{0.05}
= \frac{4.096\times 10^{-13}}{5\times 10^{-2}}
\approx 8.2\times 10^{-12}\thinspace\mathrm{m}
= 0.0082\thinspace\mathrm{nm}.
```

So a source with $1\thinspace\mathrm{nm}$ spectral span contains on the order of:

```math
N_\lambda \approx \frac{\Delta\lambda_{\mathrm{src}}}{\Delta\lambda_c} \approx \frac{1}{0.0082} \approx 1.2\times 10^2
```

independent spectral “looks.”

This is why the repo’s MMF concept puts so much emphasis on **linewidth**.

---

## 13) Time averaging at 500 µs: why 10 kHz only gives 5 patterns

If a scrambler decorrelates the speckle pattern at a rate $`f_{\mathrm{scr}}`$ (Hz), and the exposure time is $\tau$,
then the number of statistically independent patterns captured is roughly:

```math
N_t \approx f_{\mathrm{scr}}\,\tau.
```

Toy example: $`f_{\mathrm{scr}} = 10\thinspace\mathrm{kHz}`$ and $\tau = 500\thinspace\mu\mathrm{s}$:

```math
N_t \approx (10^4)\,(5\times 10^{-4}) = 5.
```

So time averaging alone gives only a small reduction: $C\to C/\sqrt{5}$.

The *whole reason* the MMF concept remains plausible at 500 µs is that you can multiply by spectral diversity:
$`N_{\mathrm{eff}} \approx N_t N_\lambda ...`$.

---

## 14) Putting it together: a concrete $500\thinspace\mu\mathrm{s}$ MMF speckle-contrast estimate

Use typical values from `configs/illumination_mmf_500us.yaml`:

- $`f_{\mathrm{scr}} = 10\thinspace\mathrm{kHz}`$ → $`N_t\approx 5`$
- $\Delta\mathrm{OPL}=5\thinspace\mathrm{cm}$ → $`\Delta\lambda_c\approx 0.0082\thinspace\mathrm{nm}`$
- suppose $`\Delta\lambda_{\mathrm{src}}=1\thinspace\mathrm{nm}`$ → $`N_\lambda\approx 122`$
- assume two polarizations effectively contribute: $`N_{\mathrm{pol}}\approx 2`$
- no angle scanning: $`N_{\mathrm{angle}}=1`$

Then:

```math
N_{\mathrm{eff}} \approx 5\times 122\times 2\times 1 \approx 1220.
```

So a naive contrast estimate is:

```math
C \approx \frac{1}{\sqrt{1220}} \approx 0.029.
```

That is “very smooth” by the standards of typical speckle.

### The caution that matters

The estimate above assumes:

- those patterns are *independent* (often only approximately true), and
- your illumination optics do not re-introduce coherent structure downstream.

This is why the repo uses simulations and sanity checks (`notebooks/07`, `09`, `10`, `11`) rather than trusting
this back-of-the-envelope alone.

---

## 15) Approach A vs Approach B in the language of equations

### Approach A (SMF): one spatial mode, but not automatically “flat”

In an ideal single-mode fiber, the field at the output is dominated by one transverse mode:

```math
U(\mathbf{r}) \approx a_0\,\psi_0(\mathbf{r})\,e^{i\phi_0}.
```

There are no intermode cross terms because there is only one mode.
So you do **not** get MMF-style speckle from intermode interference.

But you typically still get a **Gaussian** (or near-Gaussian) intensity envelope,
so “flat-field” requires beam shaping / relays / stops.

In other words:

- SMF → spatial coherence is high, but spatial structure is simple.
- uniformity is achieved by **imaging optics**, not by modal averaging.

### Approach B (MMF): many spatial modes, but can be made effectively incoherent

In an MMF, you start from:

```math
U(\mathbf{r}) = \sum_{m=1}^{M} a_m\,\psi_m(\mathbf{r})\,e^{i\phi_m}.
```

To suppress speckle you want the intensity to behave like:

```math
I(\mathbf{r}) \approx \sum_{k=1}^{N_{\mathrm{eff}}} |U_k(\mathbf{r})|^2,
```

with many mutually incoherent $`U_k`$ contributing during the exposure.

Operationally, that means forcing **decorrelation** by one (or more) of:

- time: scrambler / vibration / fast bend modulation ($`N_t`$)
- spectrum: wide linewidth or multi-line source ($`N_\lambda`$)
- polarization mixing ($`N_{\mathrm{pol}}`$)
- pupil-angle hopping ($`N_{\mathrm{angle}}`$)

The decisive inequality is:

```math
\Delta\mathrm{OPL} \gg L_c
```

because that suppresses coherent cross terms between many paths.

---

## 16) Practical checklist: what parameters you actually need to measure or bound

For the MMF concept, the “hard” unknowns are usually:

1. $`\Delta\mathrm{OPL}`$ (or equivalently $\Delta\tau$)
2. the **source linewidth** $`\Delta\lambda_{\mathrm{src}}`$
3. the **decorrelation rate** of your scrambler at the relevant mechanical mounting
4. launch conditions (how many modes are actually excited)

This repo’s workflow makes those explicit:

- Sweep $`\Delta\mathrm{OPL}`$ over plausible bounds (step-index upper bound vs graded-index reduction).
- Use vendor linewidth specs to bracket $`\Delta\lambda_{\mathrm{src}}`$.
- Translate scrambler frequency to $`N_t`$ at $500\thinspace\mu\mathrm{s}$.

If you want “the fast lane”:

- start with `configs/illumination_mmf_500us.yaml`
- run `scripts/run_mmf_500us_sweep.py`
- explore the conceptual notebook `notebooks/11_fiber_modes_speckle_interactive_3d.py`

---

## 17) Quick reference formulas (one page)

### Geometry and modes

```math
\mathrm{NA} = \sqrt{n_1^2-n_2^2}
```

```math
V = \frac{2\pi a}{\lambda}\,\mathrm{NA}
```

Single-mode (step-index fiber): $V<2.405$.

Mode count (step-index, large $V$): $M\approx V^2/2$.

### Coherence and MMF averaging

Coherence length (order):

```math
L_c \sim \frac{\lambda^2}{\Delta\lambda}
```

Step-index path-spread upper bound:

```math
\Delta\mathrm{OPL} \approx \left(\frac{\mathrm{NA}^2}{2n_1}\right)L
```

Spectral decorrelation width:

```math
\Delta\lambda_c \sim \frac{\lambda^2}{\Delta\mathrm{OPL}}
```

Independent looks:

```math
N_{\mathrm{eff}} \approx N_t N_\lambda N_{\mathrm{pol}} N_{\mathrm{angle}}
\quad\Rightarrow\quad
C \approx \frac{1}{\sqrt{N_{\mathrm{eff}}}}.
```

---

## 18) Where this plugs into the rest of the repo

- **If you want the optics-side consequences of speckle on spot detection:**
  see `docs/ILLUMINATION_SPECKLE.md`.

- **If you want “modes as shapes” (visual intuition):**
  see `notebooks/11_fiber_modes_speckle_interactive_3d.py`.

- **If you want “linewidth + OPL spread + Fourier optics” in a step-by-step derivation:**
  see `notebooks/09_mmf_wide_linewidth_scrambling_fourier_optics.py`.

- **If you want a practical “robust setup” Q&A:**
  see `notebooks/10_mmf_robust_setup_linewidth_stepindex_kohler.py`.