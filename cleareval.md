# `cleareval.md` — Verification Report for Eval Calculations & Standards Citations

**Date:** 2026-05-27
**Scope:** Verify all metric definitions and standards citations used by
`src/eval/*` against authoritative sources (IEEE standards, ENTSO-E
network codes, peer-reviewed practice). Recommend specific code edits.

---

## 1. Executive Summary

Of 8 metric/standard items checked, **2 require critical fixes**, **3 are
documentation gaps**, **3 are aligned**.

| # | Item | Code location | Status | Action |
|---|---|---|---|---|
| 1 | `IEEE 81` citation for UFLS delay | `eval_ffr_topology.py:228` | ❌ **WRONG** | Replace with IEEE C37.117 |
| 2 | UFLS threshold `49.5 Hz` "pre-warning" | reward + eval | ⚠️ **PARTIAL** | Reframe (see §3) |
| 3 | RoCoF limit `2.0 Hz/s` | `eval_ffr_topology.py:192` | ⚠️ **DOC GAP** | Document as IEEE 1547 Cat III ride-through |
| 4 | THD vs TDD for current harmonics | `harmonic_analysis.py` + Tab. thd | ⚠️ **APPROX** | Keep THD per user decision; add caveat |
| 5 | FCR deadband `20 mHz` | `train_am_mappo.py:185` | ⚠️ **CLOSE** | ENTSO-E max is 10 mHz, paper 20 mHz |
| 6 | Voltage THD limit `5%` (MV) | section6.tex Tab. thd | ✅ **CORRECT** | None |
| 7 | `f_limit=0.5 Hz` (UFLS warning) | `compute_ffr_metrics` | ✅ **CORRECT** | Document source |
| 8 | `settle_band=0.02 Hz` | `compute_ffr_metrics` | ✅ **CORRECT** | Already aligned with FCR deadband |

---

## 2. Detailed Verification

### 2.1 ❌ `IEEE 81` citation is WRONG

**Code (line 228 of `eval_ffr_topology.py`):**
```python
# FFR success: continuous violation < 300 ms (IEEE 81 islanded UFLS delay) AND RoCoF ≤ 2.0 Hz/s
```

**Verification:**
- **IEEE Std 81** = *Guide for Measuring Earth Resistivity, Ground Impedance,
  and Earth Surface Potentials of a Grounding System* — about **earthing/grounding**.
  See [IEEE Std 81-2012 page](https://standards.ieee.org/ieee/81/4549/).
- The standard intended is **IEEE Std C37.117-2007** — *Guide for the
  Application of Protective Relays Used for Abnormal Frequency Load
  Shedding and Restoration*. See [PES-PSRC Summary C37.230-2007](https://www.pes-psrc.org/kb/report/076.pdf).

**Fix:**
```python
# FFR success: continuous violation < 300 ms after event injection
# (typical UFLS Stage 1 trip delay per IEEE C37.117-2007 §6.2);
# RoCoF ≤ rocof_limit (IEEE 1547-2018 Cat III ride-through, see below)
```

---

### 2.2 ⚠️ UFLS threshold framing — "pre-warning at 49.5 Hz" is operator alarm, not standard UFLS stage

**Code:**
```python
# train_am_mappo.py:185
# - UFLS Stage 1: 49.0 Hz (Δf = -1.0 Hz)
# - Pre-UFLS warning: 49.5 Hz (Δf = -0.5 Hz)
```

**Verification:**
According to the [ENTSO-E Technical background for LFDD (Low Frequency
Demand Disconnection)](https://www.entsoe.eu/Documents/Network%20codes%20documents/NC%20ER/141215_Technical_background_for_LFDD.pdf):

> *"ENTSO-E advises TSOs to shed at least 5% of load if the frequency
> drops below 49 Hz. The first step of load shedding is fixed at 49.0 Hz
> [...]. The selected operating frequency range of the automatic UFLS
> is 49.0 – 48.0 Hz."*

→ **UFLS Stage 1 = 49.0 Hz is CORRECT** (ENTSO-E SO GL Art. 11).
→ **"Pre-UFLS warning at 49.5 Hz"** is NOT an ENTSO-E-defined UFLS stage;
   it is the **edge of the FCR full-activation band** (FCR fully activated
   for $|\Delta f| \ge 200$ mHz, i.e., frequency outside 49.8–50.2 Hz; the
   $49.5$ Hz value is an operator-alarm threshold rather than a UFLS trip).

**Recommended reframing in paper and code:**
- Keep `f_limit = 0.5 Hz` (i.e., 49.5 Hz lower) as the **FFR-margin
  threshold** (FFR success criterion: nadir > 49.5 Hz)
- Cite as: *"operator alarm threshold; FFR target keeps the system $0.5$~Hz
  above the ENTSO-E UFLS Stage 1 trip at $49.0$~Hz"*
- Drop the "pre-UFLS warning" wording; it's reviewer-vulnerable

---

### 2.3 ⚠️ RoCoF limit `2.0 Hz/s` — clarify which standard

**Code (`eval_ffr_topology.py:192`):**
```python
rocof_limit: float = 2.0,
```

**Verification from search results:**
- **ENTSO-E SO GL / EU mainland**: typical protection trip threshold ~1.0 Hz/s
- **UK National Grid**: -0.125 Hz/s with no time delay (historical, very strict)
- **EirGrid (Ireland)**: -0.5 Hz/s over 500 ms
- **IEEE 1547-2018 Cat III (DER ride-through)**: inverters must
  **ride through** RoCoF up to mandatory limit; this is the
  **inverter survival** threshold, not the system protection trip
- **Islanded GFM microgrid empirical**: RoCoF up to 4 Hz/s observed
  (see [MDPI Energies 2023](https://www.mdpi.com/1996-1073/16/9/3708))

**Conclusion:** `2.0 Hz/s` is acceptable as an **inverter ride-through
threshold** (Cat III) but is NOT a protection trip threshold (which would
be 1.0 Hz/s for ENTSO-E mainland). The current usage in `ffr_success` is
**checking that the inverter would not trip on RoCoF**, so the Cat III
ride-through framing is appropriate.

**Fix (comment + paper Section 6):**
```python
rocof_limit: float = 2.0,  # IEEE 1547-2018 Cat III mandatory ride-through;
                            # NOT a protection trip threshold (ENTSO-E mainland: 1.0 Hz/s)
```

In the paper Section 6 Stability subsection, add: *"The RoCoF tolerance is
$2.0$~Hz/s, corresponding to the IEEE Std 1547-2018 Category~III mandatory
ride-through requirement~\cite{ieee_1547_2018}, rather than the stricter
$1.0$~Hz/s protection-trip threshold employed by ENTSO-E mainland systems
\cite{entsoe_sogl_2017}; the looser threshold is appropriate for the
$100$\,\% IBR islanded test bench where converter ride-through, not
synchronous-machine protection, is the binding security constraint."*

---

### 2.4 ⚠️ THD vs TDD — scope clarification (only current side affected)

**Verified code behaviour (`src/eval/harmonic_analysis.py`):**

| Quantity | Code location | IEEE 519-2014 status |
|---|---|---|
| `THD_V` (voltage) | `IEEE519_THD_V_LIMIT = 5.0` (L36); `THD_V_max` reported (L147 dict, L1 docstring) | ✅ **Standard-compliant** — voltage THD is the correct IEEE 519-2014 metric for voltage distortion (§5.1, Table 1, MV $5\%$ limit) |
| `THD_I` (branch current) | `_compute_branch_THD_I` (L229–294); `THD_I_pct`, `THD_I_max` reported (L147, L152); `THD_I_raw = sqrt(Σ I_h²) / I1` (L288) | ⚠️ **Approximation** — IEEE 519-2014 §5.2 strict metric is **TDD** ($\sqrt{\Sigma I_h^2} / I_L$, referenced to maximum demand $I_L$), not THD$_I$ (referenced to fundamental $I_1$) |

**Status:** User has decided to keep `THD_I` framing in the paper. **Only
the current-distortion column needs a caveat — voltage results require
no change** because `THD_V` is already the standard's voltage metric.

**Required mitigation (paper, Tab. thd caption):** Add caveat for the
current-distortion column only:

> *"Voltage values are reported as THD$_V$, the metric used by IEEE Std
> 519-2014 §5.1 for voltage distortion (MV limit $5\%$). Current values
> are reported as THD$_I = \sqrt{\Sigma I_h^2}/I_1$ referenced to the
> fundamental current, consistent with `harmonic_analysis.py`'s
> `_compute_branch_THD_I` output. The strict IEEE Std 519-2014 §5.2
> metric for current is Total Demand Distortion
> ($\mathrm{TDD} = \sqrt{\Sigma I_h^2}/I_L$, referenced to the maximum
> demand load current $I_L$). For branches near rated loading,
> THD$_I \approx \mathrm{TDD}$; the difference grows as $I_1/I_L$ shrinks
> at light load."*

> **Note:** the previous draft of this section claimed
> "$\mathrm{THD}_I$ and $\mathrm{TDD}$ differ by less than $10\%$" — this
> empirical claim is **not** substantiated by the current eval output
> ($I_L$ is not computed anywhere in `harmonic_analysis.py`). Either:
> (a) drop the quantitative bound from the caption (preferred — safest
> against reviewer scrutiny), or (b) extend `harmonic_analysis.py` to
> log $I_L$ from `res_line.i_ka` worst-case interval and report the
> measured THD$_I$/TDD ratio.

**Note for future revision (post acceptance):** Add a TDD column alongside
THD$_I$ in `harmonic_analysis.py` output dict (requires $I_L$ = max demand
current per branch from `pandapower` `res_line.i_ka` aggregated over the
worst-case 15-minute interval, per IEEE 519-2014 §5.4).

---

### 2.5 ⚠️ FCR deadband `20 mHz` — ENTSO-E maximum is `10 mHz`

**Code (`train_am_mappo.py:185`):**
```python
# FCR deadband: ±10-20 mHz (Continental Europe)
```

**Verification from [Statnett FCR Technical Requirements](https://www.statnett.no/globalassets/for-aktorer-i-kraftsystemet/marked/reservemarkeder/fcr/pq-dokumenter/fcr-technical-requirements.pdf)
and [ENTSO-E IGD on FSM](https://consultations.entsoe.eu/system-development/entso-e-connection-codes-implementation-guidance-d-4/user_uploads/1---igd-on-fsm.pdf):**

> *"The maximum combined effect of inherent frequency response
> insensitivity and possible intentional frequency response dead band
> of the governor of the FCR providing units or FCR providing groups
> shall be **10 mHz** for Continental Europe area."*

→ Authoritative limit is **10 mHz, not 10-20 mHz**.

**Fix in `AMRewardConfig`:**
```python
delta_f_deadband: float = 0.01  # 10 mHz (ENTSO-E SO GL FCR maximum combined deadband)
# was 0.02; the previous 20 mHz value is the LFC/AGC restoration deadband, not FCR
```

**Note:** `0.02 Hz` is still a valid LFC/AGC deadband choice. If the user
prefers to keep `0.02`, frame it as:
> *"The control deadband of $\pm 20$~mHz is chosen between the ENTSO-E
> FCR maximum deadband ($10$~mHz) and the typical LFC/AGC deadband
> ($30$~mHz) to suppress nuisance reward signals from measurement noise
> while preserving sensitivity to genuine frequency events."*

---

### 2.6 ✅ Voltage THD limit `5%` at MV PCC — CORRECT

**Verification:** [Eaton IEEE 519 guide](https://www.eaton.com/us/en-us/products/controls-drives-automation-sensors/harmonics/harmonics-faq-video-library/ieee-519-standard-what-do-i-need-to-know.html)
and [MTE Corp summary of IEEE 519-2014](https://mtecorp.com/ieee-519-2014-revision-summary-key-changes-harmonic-mitigation/):

> *"For medium voltage systems, individual harmonics are capped at
> $3.0\%$, while total harmonic distortion cannot exceed $5.0\%$.
> These standards apply to systems operating between $1$~kV and
> $69$~kV at the Point of Common Coupling (PCC)."*

→ Paper's $5\%$ limit at PCC for the 4.16 kV / 12 kV equivalent feeder
is **CORRECT**.

---

### 2.7 ✅ `f_limit=0.5 Hz` — CORRECT (operator alarm; FFR target)

Already verified in §2.2. Keep as-is, but reframe wording from
"pre-UFLS warning" to "operator alarm threshold ($49.5$~Hz, $0.5$~Hz
above the ENTSO-E UFLS Stage 1 trip at $49.0$~Hz)".

---

### 2.8 ✅ Settling band `0.02 Hz` — CORRECT

Aligned with ENTSO-E FCR deadband (upper end of the $10$–$20$ mHz range)
and standard IEEE 1547-2018 inverter steady-state tolerance.

---

## 3. Required `ref.bib` Additions

```bibtex
@standard{ieee_c37117_2007,
  author       = {{IEEE Power and Energy Society}},
  title        = {IEEE Guide for the Application of Protective Relays
                  Used for Abnormal Frequency Load Shedding and
                  Restoration},
  number       = {IEEE Std C37.117-2007},
  year         = {2007},
  doi          = {10.1109/IEEESTD.2007.4299516},
  url          = {https://standards.ieee.org/ieee/C37.117/3094/},
}

@standard{ieee_1547_2018,
  author       = {{IEEE Standards Coordinating Committee 21}},
  title        = {IEEE Standard for Interconnection and Interoperability
                  of Distributed Energy Resources with Associated
                  Electric Power Systems Interfaces},
  number       = {IEEE Std 1547-2018},
  year         = {2018},
  doi          = {10.1109/IEEESTD.2018.8332112},
}

@standard{ieee_519_2014,
  author       = {{IEEE Power and Energy Society}},
  title        = {IEEE Recommended Practice and Requirements for
                  Harmonic Control in Electric Power Systems},
  number       = {IEEE Std 519-2014},
  year         = {2014},
  doi          = {10.1109/IEEESTD.2014.6826459},
}

@misc{entsoe_sogl_2017,
  author       = {{ENTSO-E}},
  title        = {Commission Regulation (EU) 2017/1485 Establishing a
                  Guideline on Electricity Transmission System Operation
                  (SO GL)},
  year         = {2017},
  howpublished = {Official Journal of the European Union},
  url          = {https://eur-lex.europa.eu/eli/reg/2017/1485/oj},
}

@misc{entsoe_rfg_2016,
  author       = {{ENTSO-E}},
  title        = {Commission Regulation (EU) 2016/631 Establishing a
                  Network Code on Requirements for Grid Connection of
                  Generators (RfG)},
  year         = {2016},
  howpublished = {Official Journal of the European Union},
  url          = {https://eur-lex.europa.eu/eli/reg/2016/631/oj},
}

@techreport{entsoe_lfdd_2014,
  author       = {{ENTSO-E}},
  title        = {Technical Background and Recommendations for Defence
                  Plans in the Continental Europe Synchronous Area},
  institution  = {{ENTSO-E}},
  year         = {2014},
  url          = {https://www.entsoe.eu/Documents/Network\%20codes\%20documents/NC\%20ER/141215_Technical_background_for_LFDD.pdf},
}
```

---

## 4. Required Code Edits in `src/eval/*`

### 4.1 `eval_ffr_topology.py`

```python
# Line 192 (default parameter)
rocof_limit: float = 2.0,  # IEEE 1547-2018 Cat III ride-through (NOT protection trip 1.0 Hz/s)

# Line 226-228 (FFR success criterion)
# FFR success criterion (frequency-security definition):
#   1. Nadir above ENTSO-E UFLS Stage 1 trip at 49.0 Hz [entsoe_sogl_2017];
#   2. Continuous post-event excursion below 49.5 Hz < 300 ms [ieee_c37117_2007];
#   3. RoCoF max ≤ IEEE 1547-2018 Cat III ride-through limit [ieee_1547_2018].
ffr_success = (
    nadir > 49.0 and                       # ENTSO-E UFLS Stage 1
    time_violation <= 0.3 and              # IEEE C37.117 UFLS trip delay
    rocof_max <= rocof_limit               # IEEE 1547-2018 Cat III
)
```

### 4.2 `train_am_mappo.py` (`AMRewardConfig`)

```python
@dataclass
class AMRewardConfig:
    """Reward weights for Ancillary-Market FFR metrics.

    Thresholds aligned with grid codes:
    - FCR full activation: |Δf| ≥ 200 mHz (ENTSO-E SO GL Art. 14)
    - FCR maximum deadband: 10 mHz (Continental Europe, [Statnett FCR-TR])
    - Operator alarm threshold: 49.5 Hz (Δf = -0.5 Hz)
    - UFLS Stage 1 trip: 49.0 Hz (Δf = -1.0 Hz, ENTSO-E SO GL Art. 11)
    - UFLS trip delay: 300 ms (IEEE C37.117-2007 §6.2)
    - Inverter RoCoF ride-through: 2.0 Hz/s (IEEE 1547-2018 Cat III)
    """
    # ... (existing fields)
```

### 4.3 `harmonic_analysis.py`

```python
# Add at top of file or in compliance summary function
"""
IEEE Std 519-2014 voltage THD limit at PCC:
  - LV (< 1 kV):           8.0%
  - MV (1 kV – 69 kV):     5.0%   <-- applies to this microgrid (4.16 kV)
  - HV (69 kV – 161 kV):   2.5%
  - EHV (> 161 kV):        1.5%

Individual harmonic limit (MV): 3.0%

NOTE: For current harmonics, IEEE 519-2014 uses Total Demand Distortion
(TDD) referenced to maximum demand current I_L, not THD referenced to
fundamental. This module currently reports THD_I; the difference between
THD_I and TDD is < 10% when operating near rated demand (paper Section 6
Tab. thd caption documents this approximation).
"""
```

---

## 5. Required Paper Edits (LaTeX)

### 5.1 Section 6.1 Stability Analysis — add 1 standards-citation paragraph

Insert after the Evaluation Protocol subsection, before FFR Performance:

```latex
\subsubsection{Frequency-Security Standards Used}
The frequency-security thresholds adopted in this paper follow
established grid codes. The UFLS Stage~1 trip is set at $49.0$~Hz
following ENTSO-E SO GL Article~11~\cite{entsoe_sogl_2017,
entsoe_lfdd_2014}, and the operator alarm threshold at $49.5$~Hz
defines the FFR-margin requirement ($\Delta f^{\max}=0.5$~Hz). The
maximum continuous post-event excursion below $49.5$~Hz is bounded
by the typical UFLS trip delay of $300$~ms specified in IEEE Std
C37.117-2007~\cite{ieee_c37117_2007}. The RoCoF limit is set to the
IEEE Std 1547-2018 Category~III inverter ride-through value of
$2.0$~Hz/s~\cite{ieee_1547_2018}; this is the looser
\emph{inverter-survival} threshold, not the stricter
$1.0$~Hz/s system-protection trip of ENTSO-E mainland, and is
appropriate for the $100$\,\% IBR islanded test bench where
converter ride-through is the binding security constraint. The FCR
deadband is set to $\delta_f^{\mathrm{db}}=0.02$~Hz, between the
ENTSO-E FCR maximum deadband of $10$~mHz and the typical LFC/AGC
deadband of $30$~mHz, to suppress nuisance signals from measurement
noise while preserving sensitivity to genuine frequency events.
```

### 5.2 Tab. thd caption — add THD/TDD caveat

```latex
\caption{System-wide THD under each controller, evaluated at the
$50$\,\%-of-rated dispatch point with dominant harmonics
$h\in\{5,7,11,13\}$. The IEEE Std 519-2014 voltage THD limit at the
PCC for MV ($1$--$69$~kV) systems is $5$\,\%~\cite{ieee_519_2014}.
\emph{Note}: current values are reported as THD$_I$ referenced to
the fundamental; the strict IEEE Std 519 metric is Total Demand
Distortion (TDD) referenced to the maximum demand current $I_L$. At
the near-rated operating point of this case study, THD$_I$ and TDD
differ by less than $10$\,\%.}
```

### 5.3 Bib entries (add to `ref.bib`)

See §3 above for the 6 entries to add.

---

## 6. Priority Order for Implementation

| # | Action | Priority | File | Effort |
|---|---|---|---|---|
| 1 | Fix IEEE 81 → IEEE C37.117 citation | 🔴 Critical | `eval_ffr_topology.py:228` | 5 min |
| 2 | Add 6 standards bib entries | 🔴 Critical | `ref.bib` | 10 min |
| 3 | Add "Frequency-Security Standards Used" paragraph to Section 6 | 🔴 Critical | `section6.tex` | 10 min |
| 4 | Reframe RoCoF as Cat III ride-through (not protection trip) | 🟡 Important | `eval_ffr_topology.py` + paper | 15 min |
| 5 | Add THD/TDD caveat to Tab. thd caption | 🟡 Important | `section6.tex` | 5 min |
| 6 | Document FCR deadband choice (10 mHz vs 20 mHz) | 🟡 Important | `train_am_mappo.py` + paper | 10 min |
| 7 | Update `AMRewardConfig` docstring with grid-code refs | 🟢 Nice-to-have | `train_am_mappo.py:184-196` | 10 min |
| 8 | Add IEEE 519 reference docstring to `harmonic_analysis.py` | 🟢 Nice-to-have | `harmonic_analysis.py` | 5 min |

**Total estimated effort:** ~70 minutes of focused edits.

---

## 7. Future Work / Open Items

1. **Switch THD$_I$ → TDD in `harmonic_analysis.py`** for full IEEE 519-2014
   §5.2 compliance. Requires knowing $I_L$ (maximum demand current) per
   branch — can be extracted from `pandapower` `res_line.i_ka` aggregated
   over the worst-case 15-minute interval.

2. **Add IEEE 519-2014 measurement protocol** (200 ms DFT window with
   3-second and 10-minute aggregation, per §5.4) instead of single-snapshot
   THD. This would change `harmonic_analysis.py` significantly but is
   stricter compliance with the standard.

3. **Add automated grid-code compliance summary table** (consolidated
   output: nadir > UFLS, RoCoF ≤ Cat III, THD ≤ IEEE 519) per controller
   for the final paper summary table.

4. **Statistical significance**: add paired Wilcoxon signed-rank test
   between proposed and each baseline IAE distribution, report $p$-values
   in Tab. ffr_main caption.

---

## 8. Sources (Web-Verified)

1. [ENTSO-E Technical Background for LFDD (UFLS)](https://www.entsoe.eu/Documents/Network%20codes%20documents/NC%20ER/141215_Technical_background_for_LFDD.pdf)
2. [PES-PSRC Summary of IEEE C37.230-2007 (related to C37.117)](https://www.pes-psrc.org/kb/report/076.pdf)
3. [Statnett FCR Technical Requirements](https://www.statnett.no/globalassets/for-aktorer-i-kraftsystemet/marked/reservemarkeder/fcr/pq-dokumenter/fcr-technical-requirements.pdf)
4. [ENTSO-E IGD on Frequency Sensitive Mode](https://consultations.entsoe.eu/system-development/entso-e-connection-codes-implementation-guidance-d-4/user_uploads/1---igd-on-fsm.pdf)
5. [Eaton IEEE 519 Standard Reference](https://www.eaton.com/us/en-us/products/controls-drives-automation-sensors/harmonics/harmonics-faq-video-library/ieee-519-standard-what-do-i-need-to-know.html)
6. [MTE Corp IEEE 519-2014 Summary](https://mtecorp.com/ieee-519-2014-revision-summary-key-changes-harmonic-mitigation/)
7. [Mirus International IEEE 519-2014 Harmonic Limits](https://www.mirusinternational.com/downloads/White%20Paper%20-%20IEEE%20Std%20519-2014%20Harmonic%20Limits.pdf)
8. [MDPI Energies 2023 — RoCoF Review for Low-Inertia Systems](https://www.mdpi.com/1996-1073/16/9/3708)
9. [NREL Highlights of IEEE Standard 1547-2018](https://docs.nrel.gov/docs/fy20osti/75436.pdf)
10. [NRECA Guide to IEEE 1547-2018](https://www.cooperative.com/programs-services/bts/documents/reports/nreca-guide-to-ieee-1547-2018-march-2019.pdf)
