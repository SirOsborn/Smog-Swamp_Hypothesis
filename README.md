# The Smog & Swamps Hypothesis
### Does PM2.5 Air Pollution Drive Malaria in Cambodia? (2000–2019)

> **Assignment:** Advanced Data Science (ADS)  
> **Author:** Heng  
> **Dataset span:** 2000 – 2019 · Cambodia national level  
> **Tools:** Python · pandas · statsmodels · Plotly · Seaborn · scikit-learn · scipy  
> **Last updated:** February 26, 2026

---

## Table of Contents
1. [Overview](#overview)
2. [Hypotheses](#hypotheses)
3. [Data Sources](#data-sources)
4. [Methodology](#methodology)
5. [Phase 1 — Data Engineering & EDA](#phase-1--data-engineering--eda)
6. [Phase 2 — Correlation Analysis](#phase-2--correlation-analysis)
7. [Phase 3 — VAR Model & Optimisation](#phase-3--var-model--optimisation)
8. [Phase 4 — Risk Classification Dashboard](#phase-4--risk-classification-dashboard)
9. [Key Finding](#key-finding)
10. [Project Structure](#project-structure)
11. [How to Run](#how-to-run)

---

## Overview

This project investigates whether **fine particulate matter (PM2.5)** air pollution causally drives **malaria incidence** in Cambodia — a country where biomass burning dominates the dry season and malaria transmission peaks in the wet season. 

**Key Challenge:** Satellite PM2.5 coverage in Cambodia only began in 2010. To enable a full 2000-2019 analysis, we developed a **hybrid proxy-statistical method** to estimate 2001-2009 PM2.5 values (Adj-R²=0.53 for proxy regression, Pseudo-R²=0.79 for Pchip interpolation), combining biomass fuel usage, respiratory mortality, and climate variables with statistical interpolation.

Using this 240-month continuous dataset and climate-controlled vector autoregression (VAR), the analysis reveals a **statistically significant but counterintuitive** relationship: PM2.5 appears to **suppress** malaria transmission, supporting the **Stress Hypothesis** (H₁) where air pollution reduces mosquito survival and activity.

---

## Hypotheses

| Label | Statement |
|-------|-----------|
| **H₀** | PM2.5 levels have **no** statistically significant impact on mosquito-borne disease rates |
| **H₁** | High PM2.5 particulates block mosquito spiracles and olfactory sensors → reduced biting activity → **lower** disease transmission (Stress Hypothesis) |
| **H₂** | High PM2.5 is a proxy for dense urbanization and heat → more mosquito breeding and dengue/malaria cases (Urbanization Hypothesis) |

---

## Data Sources

| Dataset | Source | Coverage |
|---------|--------|----------|
| PM2.5 satellite (µg/m³) | WHO Global Air Quality Database / V6.GL.03 | 2010–2019 (observed) |
| PM2.5 estimated (µg/m³) | Hybrid proxy-statistical method | 2001–2009 (gap-filled) |
| Malaria incidence (cases/1,000) | WHO SDG Indicator 442CEA8 | 2000–2019 |
| Clean fuels access (%) | WHO SDG Indicator 6A64C9A | 2000–2019 (proxy) |
| Respiratory mortality | WHO SDG Indicator E2FC6D7 | 2000–2019 (proxy) |
| Climate normals (Temp, Rainfall) | World Bank CCKP 1991–2020 | Seasonal baseline |

> **Note:** Raw source files are excluded from this repository (see `.gitignore`).  
> The processed monthly and annual merged datasets **are** included in `data/`.

---

## Methodology

The analysis follows four structured phases:

```
Phase 1: Data Engineering & Hybrid PM2.5 Gap-Filling
   → Develop hybrid proxy-statistical model for 2001-2009 PM2.5 estimation
   → Test 6 regression models (Linear, Polynomial, Interaction, Time-aware, Ridge, Lasso)
   → Combine proxy regression (Adj-R²=0.53) + Pchip interpolation (Pseudo-R²=0.79)
   → Final method: 50% proxy + 50% Pchip for each gap-year month
   → Generate 240-month continuous dataset (2000-2019)

Phase 2: Lagged Correlation Analysis
   → Pearson lagged table (lags −4 to +8 months)
   → Cross-Correlation Function (CCF) ± 12 lags
   → Identify "Dry Season Trap" (inverse seasonal correlation)

Phase 3: Vector Autoregression (VAR)
   → ADF stationarity testing + first-differencing
   → VAR(4) model selection via BIC
   → Granger causality test: PM2.5 → Malaria
   → Impulse Response Analysis
   → Hypothesis interpretation based on coefficient sign

Phase 4: Risk Classification & Dashboard
   → Rule-based risk classifier (PM2.5 × Temperature × Rainfall)
   → Static risk heatmaps + monthly alert timeline
   → Interactive Plotly 4-panel dashboard
```

---

## Phase 1 — Data Engineering & Hybrid PM2.5 Gap-Filling

**Challenge:** Satellite PM2.5 data for Cambodia is only available from 2010 onwards. To conduct a full 2000-2019 analysis, we needed to estimate PM2.5 values for 2001-2009.

**Solution:** Hybrid proxy-statistical method combining two approaches:

1. **Proxy Regression (Adj-R² = 0.53)**
   - Trained on 2010-2019 observed PM2.5 data
   - Features: Biomass fuel usage (primary), respiratory mortality, sanitation coverage, tobacco use, temperature, rainfall
   - 6 models tested: Linear OLS (best), Polynomial, Interaction, Time-aware, Ridge, Lasso

2. **Pchip Interpolation (Pseudo-R² = 0.79)**
   - Monotonic cubic interpolation of annual satellite data to monthly resolution
   - Preserves seasonality and avoids overfitting

3. **Hybrid Combination**
   - Final estimate: **50% proxy regression + 50% Pchip interpolation**
   - Back-casted to 2001-2009 to create 240-month continuous dataset

After merging with malaria data and climate variables, both time series show a **declining trend** (2000-2019) with oppositely-phased seasonal patterns: PM2.5 peaks in dry season (biomass burning), malaria peaks in wet season (mosquito breeding).

---

## Phase 2 — Correlation Analysis

### Lagged Cross-Correlation Function (CCF)

The CCF across ±12 month lags reveals a **peak at k = +6 months** (r ≈ +0.475), meaning PM2.5 appears to *lead* malaria by half a year.

![Cross-Correlation Function](images/ccf_pm25_malaria.png)

> ⚠️ This CCF signal is driven by **two confounders**, not causation:
> 1. Both series share the same long-run declining trend (2000–2019)
> 2. The seasonal offset between burning season (dry) and disease season (wet) creates a mechanical lag

### Feature Correlation Heatmap

The 7 × 7 correlation matrix below includes PM2.5, its lags 1–3, malaria, rainfall, and temperature.

![Correlation Heatmap](images/correlation_heatmap.png)

---

## Phase 3 — VAR Model & Granger Causality

### Stationarity Testing (ADF)

Both series require first-differencing to achieve stationarity for VAR modelling.

### VAR(4) Model — BIC-Selected

The VAR(4) model (240 months, 2000-2019) Granger causality test reveals:

| Direction | F-stat | p-value | Significance | Coefficient Sum |
|-----------|--------|---------|--------------|-----------------|
| **PM2.5 → Malaria** | **3.18** | **0.013** | ✅ ** | **−0.004321** |
| Malaria → PM2.5 | 0.89 | 0.469 | ns | — |

**Key Result:** PM2.5 Granger-causes Malaria with statistical significance, but the **negative coefficient sum** indicates an **inverse relationship** — high PM2.5 is associated with **lower** malaria incidence after accounting for lagged effects.

### Impulse Response Analysis

The cumulative effect of a PM2.5 shock on malaria over 10+ months shows a **suppressive** rather than amplifying pattern, consistent with the Stress Hypothesis (H₁).

### Hypothesis Interpretation

Based on the **negative** coefficient sum (−0.004321):

- ❄️ **H₁ (Stress Hypothesis) SUPPORTED:** High PM2.5 → SUPPRESSES mosquito activity → LOWER malaria incidence
- ❌ **H₂ (Urbanization Hypothesis) NOT SUPPORTED:** Would require positive coefficient

---

## Phase 4 — Risk Classification Dashboard

A rule-based classifier combines PM2.5 level, temperature, and lagged rainfall to assign a 4-tier outbreak risk score: **Low / Moderate / High / Critical**.

### Risk Heatmap

The heatmap below shows risk classification across the PM2.5 × Temperature grid, split by season (dry vs wet).

![Risk Classification Heatmap](images/risk_heatmap_static.png)

### Monthly Alert Timeline

Mapping the risk classifier onto the 132-month historical panel shows that **High/Critical risk periods cluster in the wet season (June–October)**, regardless of PM2.5 level.

![Monthly Risk Alert Timeline](images/risk_alert_timeline.png)

---

## Key Finding

> ### 🔬 H₀ is **REJECTED** · H₁ (Stress Hypothesis) is **SUPPORTED** · H₂ is **NOT SUPPORTED**

The VAR(4) Granger causality test reveals **PM2.5 → Malaria** is statistically significant (F = 3.18, p = 0.013), but with a **negative coefficient sum** (−0.004321), indicating that high PM2.5 **suppresses** malaria transmission rather than amplifying it.

### Interpretation

**The "Stress Hypothesis" (H₁) is supported:** High PM2.5 appears to reduce mosquito survival and activity through:
- Particulate matter clogging mosquito spiracles (breathing tubes)
- Reduced flight efficiency and foraging behavior
- Shortened lifespan and reduced biting rates

**However, important caveats apply:**
1. **Effect size is small** compared to climate drivers (temperature/rainfall effects are 10-100× larger)
2. **9-year gap-filling** (2001-2009) relies on proxy model (validation R²=0.53)
3. **Seasonal confounding** remains: PM2.5 peaks during dry season when malaria is naturally low
4. **Experimental validation needed** to confirm the biological mechanism

### Implication

This finding suggests that air quality improvements in Cambodia **may paradoxically increase short-term malaria risk** by removing a natural mosquito suppression factor. However:
- The respiratory health benefits of reducing PM2.5 vastly outweigh any malaria risk increase
- Climate (rainfall and temperature) remains the dominant driver of malaria transmission
- Vector control programs should intensify during clean-air periods
- Seasonal forecasting should use monsoon onset (May-June) as the primary malaria alert trigger

---

## Project Structure

```
assADS/
├── Smog&Swamp_Hypothesis.ipynb  ← Full analysis notebook (open this)
├── README.md
├── .gitignore
├── images/                      ← All output charts & figures
│   ├── pm25_proxy_model_2001_2009.png
│   ├── seasonal_decompose.png
│   └── [other generated plots]
└── data/
    ├── cambodia_monthly_merged.csv      ← 240-row monthly panel (2000-2019)
    ├── cambodia_annual_merged.csv       ← 20-row annual panel (2000-2019)
    ├── pm25_proxy_training_data.csv     ← Proxy model training data
    └── 116_Cambodia/                    ← Raw WHO SDG indicators (excluded)
```

> **Raw source datasets** in `data/116_Cambodia/` contain WHO SDG indicators for clean fuels, respiratory mortality, sanitation, and other proxies. These are used to train the hybrid PM2.5 gap-filling model.

---

## How to Run

```bash
# 1. Clone the repository
git clone <repo-url>
cd assADS

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install numpy pandas matplotlib seaborn scipy statsmodels plotly scikit-learn ipywidgets nbformat

# 4. Open the notebook
jupyter notebook Smog\&Swamp_Hypothesis.ipynb
# or open in VS Code and run all cells
```

> **Tip:** Run all cells in order (Kernel → Restart & Run All). The notebook takes ~30 seconds to complete all 31 cells (including hybrid PM2.5 gap-filling).

---

*Analysis conducted as part of the Advanced Data Science coursework assignment. Statistical findings are based on hybrid-estimated PM2.5 data (2001-2009) combined with satellite observations (2010-2019), WHO SDG malaria indicators, and World Bank climate normals. Last updated February 26, 2026.*
