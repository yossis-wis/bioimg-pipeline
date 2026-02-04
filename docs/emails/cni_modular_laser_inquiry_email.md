# Draft email: quote request for 640 + 488 nm laser modules (Approach A vs B)

> **Editing note:** **the email can be trivially shortened (while keeping the diagrams + questions intact)** by deleting the sections **“Background / funding constraint”** and **“Rationale / context”** and keeping everything from **“Request (action needed)”** onward.

---

## Email draft (copy/paste)

**To:** Ori Linenberg (Roshel Electroptics) <ori@roshelop.co.il>; CNI sales / Demi Zhang <trade@cnilaser.com>  
**Cc:** (optional) Gennadi <gennadi@roshelop.co.il>  
**Subject (suggested):** Quote request: 640 + 488 nm modules (SM-fiber vs wide-linewidth) — model, price, delivery to Israel

Hi Ori, hi Demi,

### Background / funding constraint (optional)
Unfortunately, our **US$50,000 departmental equipment grant was not funded**.

I now have **~10 days** to submit a **new grant application (up to US$25,000)**. If awarded, the funds would be available **by the end of this month**.

### Rationale / context (optional, but helpful)
To reduce **up-front cost** and **lead time**, I’d like to purchase in **modules** and start with **only two wavelengths**:

- **640 nm**
- **488 nm**

At this stage we do **not** need AOMs or fast gating; **CW output is fine**.

We are considering two illumination architectures:

- **Approach A (single-mode fibers):** each wavelength is a **TEM00 (or near) source** with its **own SM fiber-coupled output** (FC/APC), then we **combine beams in free space**.
- **Approach B (wide linewidth + MMF concept):** use **wide-linewidth free-space lasers** intended to be combined into a **single multimode fiber (MMF)** + scrambler.

### Request (action needed)
Could you please reply with the following for each option below:

- **CNI model name / number** you recommend
- **Price** (laser + PSU; itemized is fine)
- **Total delivery time** from order placed until the item is **received at Weizmann (Israel)**, including:
  - **build/ship lead time**
  - **typical shipping duration**

**Option A: SM fiber-coupled outputs (FC/APC)**
- **640 nm:** ~500 mW, TEM00/near, **SM fiber-coupled**, FC/APC
- **488 nm:** ~50 mW, TEM00/near, **SM fiber-coupled**, FC/APC

**Option B: wide-linewidth free-space outputs (for MMF illumination concept)**
- **640 nm:** ~1–2 W, **wide linewidth** (roughly Δλ ~2–20 nm), **free-space**
- **488 nm:** ~100 mW, **wide linewidth** (≥ ~2 nm), **free-space**

If you have a **clearly faster-delivery and/or lower-cost** variant that still fits these rough specs, please propose it.

---

## Diagrams (for reference)

### Approach A diagram (single-mode fiber per wavelength, free-space combination)

![Approach A: single-mode fiber](assets/approach_A_single_mode_fiber.png)

```text
[640 nm SM fiber-coupled] --FC/APC--> [fiber collimator] --\
                                                         >-- [dichroic combiner] --> [to objective]
[488 nm SM fiber-coupled] --FC/APC--> [fiber collimator] --/
```

### Approach B diagram (wide-linewidth lasers, intended for MMF illumination concept)

![Approach B: wide linewidth + MMF](assets/approach_B_multimode_fiber.png)

```text
[640 nm wide-linewidth] --\
                           >-- [combine] --> [MMF coupler] --> [MM fiber + scrambler] --> [to objective]
[488 nm wide-linewidth] --/
```

---

## Questions for CNI (please answer as briefly as possible)

1) **Which model(s)** do you recommend for each of the four laser modules above?
2) **What is the unit price** for each (laser + PSU)?
3) **Delivery timeline to Israel:** estimated time from order placed to receipt (build/ship lead time + shipping time).

---

Warmly,  
Yossi / Joseph Steinberger  
Weizmann Institute of Science
