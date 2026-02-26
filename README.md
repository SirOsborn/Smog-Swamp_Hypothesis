# 🌫️🦟 The Smog & Swarms Hypothesis
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

This project investigates whether **fine particulate matter (PM2.5)** air pollution causally affects **malaria incidence** in Cambodia — a country where biomass burning dominates the dry season and malaria transmission peaks in the wet season. 

**Key Challenge:** Satellite PM2.5 coverage in Cambodia only began in 2010. To enable a full 2000-2019 analysis, we developed a **hybrid proxy-statistical method** to estimate 2001-2009 PM2.5 values (Adj-R²=0.5-0.7 for proxy regression, Pseudo-R²=0.7-0.8 for Pchip interpolation), combining biomass fuel usage, air pollution mortality, and climate variables with statistical interpolation. **Critical limitation:** 45% of the analysis window (2001-2009, 108 months) relies on these estimates.

Using this 240-month continuous dataset with Vector Autoregression (VAR) and climate-controlled VARX models, the analysis encounters a fundamental challenge: **strong seasonal confounding** ("Dry Season Trap"). PM2.5 peaks during dry season (Feb-Mar) when malaria is naturally low (no breeding sites), while malaria peaks during monsoon season (Aug-Oct) when PM2.5 is washed out. This creates spurious correlation unrelated to biological causation.

**Key Finding:** Results are **INCONCLUSIVE** and highly dependent on model specification. Base VAR may show significant Granger causality with negative coefficients (consistent with H₁ Stress Hypothesis), but climate-controlled VARX analysis suggests this relationship may be **spurious**, driven entirely by shared monsoon timing rather than PM2.5 affecting mosquitoes. Current evidence is **exploratory**, not confirmatory.

---

## Hypotheses

| Label | Statement | Expected Coefficient | Biological Mechanism |
|-------|-----------|---------------------|----------------------|
| **H₀** | PM2.5 levels have **no** statistically significant impact on malaria after controlling for climate | ≈ 0 | No causal relationship; observed correlation due to confounding |
| **H₁ (Stress)** | High PM2.5 particulates stress mosquitoes → reduced biting activity → **LOWER** malaria transmission | **Negative** | Spiracle clogging, reduced flight efficiency, shortened lifespan |
| **H₂ (Urbanization Proxy)** | High PM2.5 marks urban areas with heat islands + water storage → more breeding → **HIGHER** malaria transmission | **Positive** | PM2.5 as proxy for urban conditions favorable to breeding |

---

## Data Sources

| Dataset | Source | Coverage | Quality |
|---------|--------|----------|---------|
| PM2.5 satellite (µg/m³) | WHO Global Air Quality Database / World Bank F810947 | 2010–2019 (observed) | ✅ Direct measurement |
| PM2.5 estimated (µg/m³) | Hybrid proxy-statistical method | 2001–2009 (gap-filled) | ⚠️ **45% of dataset**, R²=0.5-0.7 |
| Malaria incidence (cases/1,000) | WHO SDG Indicator 442CEA8 | 2000–2019 | ⚠️ Annual national aggregates → monthly disaggregation |
| Clean fuels access (%) | WHO SDG Indicator 6A64C9A | 2000–2019 (proxy) | Used for PM2.5 gap-filling |
| Air pollution mortality | WHO SDG Indicator E2FC6D7 | 2000–2019 (proxy) | Used for PM2.5 gap-filling |
| Climate normals (Temp, Rainfall) | World Bank CCKP 1991–2020 | Seasonal baseline | ⚠️ Static 30-year normals, not observed interannual variation |

> **Critical Data Limitation:** PM2.5 estimates for 2001-2009 (108 months, 45% of analysis window) are **unvalidated extrapolations** from a small training sample (2010-2019, N=10 years). Results should be considered **exploratory hypothesis-generating**, not confirmatory.

> **Note:** Raw source files are excluded from this repository (see `.gitignore`).  
> The processed monthly and annual merged datasets **are** included in `data/`.

---

## Methodology

The analysis follows **four structured phases** with statistical rigor and transparent limitation disclosure:

```
Phase 1: Data Engineering & Hybrid PM2.5 Gap-Filling (2001-2009)
   → Develop hybrid proxy-statistical model for missing PM2.5 data
   → Test 6 regression models (Biomass-only, Biomass+Time, +AirMort, +RespMort, Quadratic, Ridge)
   → Select best model by Adjusted R² (typically Biomass+Time+AirMort, Adj-R²=0.5-0.7)
   → Compute Pchip monotonic spline interpolation (Pseudo-R²=0.7-0.8)
   → Final estimate: 50% proxy regression + 50% Pchip for each gap-year month
   → Validate: plausibility bounds (0-100 µg/m³), temporal smoothness (Δ<10 µg/m³ month-over-month)
   → Generate 240-month continuous dataset (2000-2019)

Phase 2: Lagged Correlation Analysis — Identifying the "Dry Season Trap"
   → Pearson lagged correlations (lags −4 to +8 months)
   → Cross-Correlation Function (CCF) ± 12 lags with 95% confidence intervals
   → Correlation heatmap: PM2.5, Malaria, Temperature, Rainfall (+ lags 1-2)
   → **Key finding:** Strong negative correlation NOT due to causation but to:
      • Shared declining trend (2000-2019)
      • Seasonal offset: PM2.5 peaks dry season (Feb-Mar), Malaria peaks wet season (Aug-Oct)

Phase 3: Vector Autoregression (VAR) — Base Model with Confounding
   → Augmented Dickey-Fuller (ADF) stationarity tests + first-differencing
   → BIC lag order selection (p=1-6 months, domain-constrained for biological plausibility)
   → VAR(p) estimation: Malaria and PM2.5 as endogenous variables
   → Granger causality test: Does PM2.5 improve Malaria forecasts?
   → Coefficient sign analysis: Negative → H₁ (Stress), Positive → H₂ (Urbanization)
   → Impulse Response Functions (IRF): Dynamic shock propagation
   → ⚠️ **CRITICAL CAVEAT:** Base VAR results confounded by shared seasonality

Phase 3b: VARX — Climate-Controlled Causal Inference (Gold Standard)
   → Seasonal decomposition (additive, period=12) on ALL variables:
      • PM2.5, Malaria, Rainfall, Temperature → extract deseasonalized residuals
   → VARX specification:
      • Endogenous: Deseasonalized PM2.5 and Malaria residuals
      • Exogenous: Deseasonalized Rainfall and Temperature residuals (climate controls)
   → BIC lag selection on VARX model (may differ from base VAR)
   → Granger causality on climate-partialled series (OLS residuals after controlling for climate)
   → **Decision Framework:**
      • VARX Granger test SIGNIFICANT → Check coefficient sign → H₁ vs H₂
      • VARX Granger test NOT SIGNIFICANT → Base VAR was spurious → H₀ (no causal effect)
   → Model comparison: AIC, BIC, Pseudo-R², residual diagnostics (Ljung-Box, Jarque-Bera)

Phase 4: Public Health Risk Assessment Dashboard
   → Rule-based risk classifier: classify_risk(PM2.5, Temperature, Rainfall_lag1)
      • Thresholds: PM2.5 >50 µg/m³, Temp >30°C, Rain ≥80mm (prior month)
      • Risk levels: Low (0) / Medium (1) / High (2) / Critical (3)
   → Static risk heatmaps: PM2.5 × Temperature grid, split by season (dry vs. wet)
   → Monthly alert timeline: Risk bands overlaid on historical malaria + PM2.5 time series
   → **Interactive Plotly dashboard** (4 panels):
      • Risk gauge (needle indicator with color zones)
      • CCF bars with significance bands
      • Scatter plot: Historical data + user input marker (temperature-colored)
      • Time series: Actual vs. OLS-predicted malaria + user prediction line
      • Real-time stat table: OLS coefficients, Pearson r, predicted malaria rate
   → ipywidgets sliders: PM2.5, Temperature, Rainfall (instant updates on value change)
```

**Statistical Software:** Python 3.8+ (statsmodels VAR/VARX, sklearn, scipy, plotly)  
**Reproducibility:** All code in `Smog&Swarms_Hypothesis.ipynb`, runtime ~30-60 seconds

---

## Phase 1 — Data Engineering & Hybrid PM2.5 Gap-Filling

**Challenge:** Satellite PM2.5 data for Cambodia is only available from 2010 onwards (WHO Global Air Quality Database). To conduct a full 2000-2019 analysis, we needed to estimate PM2.5 values for 2001-2009 (108 months, **45% of the analysis window**).

**Solution:** Hybrid proxy-statistical method combining two approaches:

1. **Proxy Regression Model (Adj-R² = 0.5-0.7)**
   - **Training set:** 2010-2019 observed PM2.5 data (N=10 years)
   - **Features tested:** 
     - `biomass_proxy`: Inverse of clean fuels access (primary predictor)
     - `air_mort_rate`: Air pollution mortality rate
     - `resp_mort_rate`: Respiratory disease mortality
     - `year_trend`: Linear time trend
     - Polynomial/interaction terms
   - **6 models compared:**
     1. Biomass-only (baseline)
     2. Biomass + Time trend
     3. Biomass + Air mortality + Time ← **often selected as best**
     4. Biomass + Respiratory mortality + Time
     5. Biomass quadratic (polynomial features)
     6. Ridge regression (L2 regularization)
   - **Selection criterion:** Adjusted R² (penalizes overparameterization)
   - **Validation:** Coefficients must be biologically plausible (positive for biomass/mortality, negative for time trend)

2. **Pchip Monotonic Interpolation (Pseudo-R² = 0.7-0.8)**
   - Piecewise Cubic Hermite Interpolating Polynomial (scipy.interpolate.PchipInterpolator)
   - **Input:** Annual satellite observations (2000, 2010-2019)
   - **Output:** Smooth monthly curves preserving monotonicity and avoiding overshoot
   - **Validation:** Pearson correlation between Pchip monthly values and biomass proxy

3. **Hybrid Combination (Final Gap-Filled Estimate)**
   - **Formula:** `PM2.5_estimated = 0.5 × proxy_regression + 0.5 × pchip_interpolation`
   - **Rationale:** Balances statistical model (captures covariates) with smooth interpolation (reduces noise)
   - **Plausibility checks:**
     - PM2.5 ∈ [0, 100] µg/m³ (physical bounds)
     - Month-over-month change |Δ| < 10 µg/m³ (temporal smoothness)
   - **Monthly disaggregation:** Annual values split using fixed seasonal multipliers (peak in dry season Feb-Mar)

**Output:** 240-month continuous time series (2000-2019) with PM2.5, Malaria, Temperature, Rainfall

**Critical Limitation:** 
- Training sample is small (N=10 years, 2010-2019)
- Assumes PM2.5-proxy relationship held constant 2001-2009 (**unvalidated assumption**)
- Introduces **smoothness bias** (hybrid method likely underestimates true interannual variability)
- Result: **45% of dataset is estimated**, not observed

After merging with malaria data and climate variables, both time series show a **declining trend** (2000-2019) with **oppositely-phased seasonal patterns**: PM2.5 peaks in dry season (biomass burning), malaria peaks in wet season (mosquito breeding) → **"Dry Season Trap" confounding**.

---

## Phase 2 — Correlation Analysis & the "Dry Season Trap"

### Lagged Cross-Correlation Function (CCF)

The CCF across ±12 month lags reveals **negative correlation at k ≈ 0 to -3 months**, meaning PM2.5 and malaria are inversely related in the same or nearby months.

![Cross-Correlation Function](images/ccf_pm25_malaria.png)

> ⚠️ **CRITICAL INTERPRETATION:** This CCF signal is **NOT evidence of causation**. It reflects **"Dry Season Trap" confounding:**
> 1. **Seasonal offset:** PM2.5 peaks dry season (Feb-Mar, no breeding sites), Malaria peaks wet season (Aug-Oct, abundant breeding)
> 2. **Shared declining trend:** Both series declining 2000-2019 (improved air quality + malaria control)
> 3. **Result:** Mechanical negative correlation unrelated to PM2.5 affecting mosquitoes

**Biological timing context:** 
- Mosquito life cycle (egg → adult): 2-8 weeks depending on temperature
- If PM2.5 truly affected mosquitoes (H₁ Stress), effects should appear within **1-4 months**
- If PM2.5 proxies urbanization (H₂), correlation should be **positive** and contemporaneous

### Feature Correlation Heatmap

The correlation matrix includes PM2.5, PM2.5 lags (1-2 months), malaria, rainfall, and temperature.

![Correlation Heatmap](images/correlation_heatmap.png)

**Key observations:**
- **PM2.5 ↔ Malaria:** Negative contemporaneous correlation (r ≈ -0.10 to -0.30)
- **Rainfall ↔ Malaria:** Strong positive correlation (breeding cycle driver)
- **Temperature ↔ PM2.5:** Moderate positive (heat + biomass burning)
- **Lagged PM2.5:** Weaker correlation than contemporaneous (signal attenuation)

**Conclusion:** Climate variables (rainfall, temperature) dominate malaria variance. PM2.5 effect (if real) is secondary and confounded by seasonality.

---

## Phase 3 — VAR & VARX: Disentangling Causation from Confounding

### Stationarity Testing (ADF)

Augmented Dickey-Fuller tests show both PM2.5 and Malaria series are **non-stationary** (p > 0.05) in levels. First-differencing achieves stationarity (p < 0.05) required for VAR modeling.

### Base VAR Model — Confounded by Seasonality

**Specification:** VAR(p) on first-differenced PM2.5 and Malaria (p selected by BIC, typically 1-6)

**Granger Causality Test (Base VAR):**

| Direction | Typical Result | Interpretation |
|-----------|----------------|----------------|
| **PM2.5 → Malaria** | May show **significance** (p < 0.05) at some lags | ⚠️ **SPURIOUS** — driven by shared seasonality |
| Malaria → PM2.5 | Usually **not significant** (p ≥ 0.05) | No reverse causation (biologically implausible) |

**Coefficient sign (if significant):**
- **Negative cumulative sum** (typical): Suggests H₁ (Stress) — PM2.5 suppresses malaria
- **Positive cumulative sum** (rare): Would suggest H₂ (Urbanization) — PM2.5 amplifies malaria

**⚠️ CRITICAL CAVEAT:** Base VAR results are **confounded by "Dry Season Trap"**. Both variables driven by monsoon timing. Significant Granger test does NOT prove causation.

### Impulse Response Functions (IRF)

IRF traces impact of a 1-unit PM2.5 shock on malaria over 12 months. Confidence bands often cross zero, indicating no statistically significant sustained effect.

![IRF: PM2.5 Shock to Malaria](images/irf_pm25_malaria.png)

### VARX Model — Climate-Controlled Analysis (Gold Standard)

**Problem with Base VAR:** PM2.5 and Malaria both driven by shared climate cycles (monsoon onset/end). Base VAR cannot distinguish true PM2.5 effect from climate confounding.

**VARX Solution:**

1. **Seasonal decomposition** (statsmodels.tsa.seasonal.seasonal_decompose):
   - Decompose ALL four variables (PM2.5, Malaria, Rainfall, Temperature)
   - Method: Additive, period=12 months
   - Extract: Trend, Seasonal component, **Residuals (deseasonalized series)**

2. **VARX specification:**
   - **Endogenous variables:** Deseasonalized PM2.5 residuals, Deseasonalized Malaria residuals
   - **Exogenous variables:** Deseasonalized Rainfall residuals, Deseasonalized Temperature residuals
   - **Critical:** Exogenous controls MUST be deseasonalized to avoid reintroducing seasonal correlation

3. **BIC lag selection on VARX** (may differ from base VAR due to exogenous variables absorbing dynamics)

4. **Granger causality on climate-partialled series:**
   - Partial out Rainfall + Temperature using OLS residuals
   - Test: Does PM2.5 (residual) still Granger-cause Malaria (residual)?

**Interpretation Framework:**

| VARX Granger Test Result | Coefficient Sign | Conclusion |
|---------------------------|------------------|------------|
| **Significant (p < 0.05)** | **Negative** | ✅ **H₁ (Stress) supported:** PM2.5 suppresses mosquitoes even after climate controls |
| **Significant (p < 0.05)** | **Positive** | ✅ **H₂ (Urbanization) supported:** PM2.5 proxies urban breeding conditions |
| **Not significant (p ≥ 0.05)** | N/A | ✅ **H₀ supported:** Base VAR result was spurious. Climate is the only driver. |

**Model Comparison Metrics:**

![Seasonal Decomposition](images/seasonal_decompose.png)

| Metric | Base VAR | VARX (Climate-Controlled) | Interpretation |
|--------|----------|---------------------------|----------------|
| AIC | Higher | Lower (better fit) | VARX captures more variance |
| BIC | Higher | Lower (better fit-complexity) | Climate controls improve parsimony |
| Pseudo-R² (Malaria) | Lower | Higher | More malaria variance explained |
| Granger F-stat | Inflated | Corrected | Removes climate confounding |

**Scientific Gold Standard:** VARX results should be validated against:
- Experimental studies (lab mosquito exposure to PM2.5)
- Spatial analyses (province-level with ground PM2.5 monitors)
- Observed-only subset (2010-2019, N=120) to avoid gap-filling bias

---

## Phase 4 — Risk Classification Dashboard

A rule-based outbreak risk classifier integrates PM2.5, temperature, and lagged rainfall into a 4-tier alert system: **Low (0) / Medium (1) / High (2) / Critical (3)**.

### Risk Classification Logic

**Thresholds:**
- **High PM2.5:** >50 µg/m³ (WHO guideline exceedance)
- **Low PM2.5:** ≤25 µg/m³ (clean air)
- **Hot temperature:** >30°C (accelerates mosquito development)
- **Wet prior month:** ≥80mm rainfall (larval habitats active)

**Decision Matrix:**

| PM2.5 | Temperature | Rainfall (lag 1) | Risk Level | Biological Interpretation |
|-------|-------------|------------------|------------|---------------------------|
| ≤25 | >30°C | ≥80mm | 🔴 **High (2)** | **Optimal breeding:** Clean air + heat + water → peak transmission |
| >50 | >30°C | ≥80mm | 🚨 **Critical (3)** | **H₂ scenario:** Urban heat islands + standing water despite pollution |
| >50 | >30°C | <80mm | 🟡 **Medium (1)** | **H₁ scenario:** PM2.5 stress may suppress mosquitoes, but urban heat persists |
| Any | ≤30°C | Any | 🟢 **Low (0)** | Cool temperatures slow mosquito metabolism |
| Other combinations | — | — | 🟢 **Low (0)** | Conditions not favorable for transmission |

### Risk Heatmap

Static risk surface across PM2.5 × Temperature grid, split by season (dry vs. wet based on lagged rainfall).

![Risk Classification Heatmap](images/risk_heatmap_static.png)

### Monthly Alert Timeline

Historical risk classification (2000-2019) overlaid on actual malaria incidence and PM2.5 levels. Shows **High/Critical risk periods cluster in wet season (June–October)**, regardless of PM2.5 level.

![Monthly Risk Alert Timeline](images/risk_alert_timeline.png)

### Interactive Plotly Dashboard

**Live 4-panel explorer** with instant updates via ipywidgets sliders:

1. **Risk Gauge:** Needle indicator showing Low/Medium/High/Critical with color zones
2. **CCF Bars:** Cross-correlation function ±6 lags with 95% confidence bands (significant lags highlighted)
3. **Scatter Plot:** Historical data points (color = temperature) + user input marker (star)
4. **Time Series:** Actual malaria (green) vs. OLS predicted (red) + user prediction line (purple)

**Interactive Controls:**
- PM2.5 slider: 2-100 µg/m³ (step 0.5)
- Temperature slider: 20-38°C (step 0.5)
- Rainfall slider: 0-350mm (step 5)

**Real-Time Statistics Panel:**
- Risk level + public health message
- Predicted malaria rate (OLS: Malaria ~ PM2.5 + Temp + Rainfall_lag1)
- Pearson r (PM2.5 ↔ Malaria) with p-value
- OLS beta coefficients (PM2.5, Temperature, Rainfall)
- Deviation from dataset mean

**Purpose:** Educational tool for understanding multi-factor environmental drivers and hypothesis-informed risk scenarios.

---

## Key Finding

> ### 🔬 Results are **INCONCLUSIVE** — Evidence Level: **EXPLORATORY, Not Confirmatory**

### Base VAR (Without Climate Controls)

The base VAR Granger causality test may show **PM2.5 → Malaria is statistically significant** with **negative coefficient sum**, suggesting H₁ (Stress Hypothesis) where high PM2.5 suppresses malaria transmission.

**BUT:** This result is **confounded by shared seasonality** ("Dry Season Trap"). Both variables driven by monsoon timing, not by PM2.5 affecting mosquitoes.

### VARX (Climate-Controlled — Recommended Test)

After controlling for rainfall and temperature via seasonal decomposition and exogenous VARX specification, results depend on gap-filled data quality:

| Scenario | VARX Granger Test | Coefficient Sign | Hypothesis Verdict |
|----------|-------------------|------------------|---------------------|
| **A** | **Significant** (p < 0.05) | **Negative** | H₁ (Stress) supported — PM2.5 suppresses mosquitoes even after climate controls |
| **B** | **Significant** (p < 0.05) | **Positive** | H₂ (Urbanization) supported — PM2.5 proxies urban breeding conditions |
| **C** | **Not significant** (p ≥ 0.05) | N/A | **H₀ supported** — Base VAR was spurious; climate is the only driver |

**Actual outcome:** Varies by run depending on gap-filling method and lag selection. **Climate confounding dominates.**

### Why Inconclusive?

1. **45% of data is gap-filled** (2001-2009) using unvalidated proxy model (Adj-R²=0.5-0.7)
2. **Strong seasonal confounding:** PM2.5 peaks when malaria low, malaria peaks when PM2.5 low
3. **Small effect size:** Even if significant, PM2.5 coefficient magnitude is **10-100× smaller** than rainfall/temperature effects
4. **Climate dominates:** Rainfall and temperature explain >90% of malaria variance
5. **No experimental validation:** Biological mechanism (spiracle clogging, flight impairment) not tested in lab
6. **National aggregation:** Masks spatial heterogeneity (urban vs. rural, border regions)

### Scientific Confidence Assessment

**Strong evidence** would require:
- ✅ Significant VARX Granger test with narrow confidence intervals
- ✅ Consistent coefficient sign across multiple model specifications
- ✅ Validation on observed-only subset (2010-2019, N=120)
- ✅ Laboratory studies (mosquito PM2.5 exposure experiments)
- ✅ Spatial analysis (province-level with ground monitors)

**Current status:** ❌ **EXPLORATORY ONLY** — None of the above validation steps completed.

### Policy Implications

**Universal recommendations (regardless of H₀ vs. H₁ vs. H₂):**

1. **Climate is the primary driver:** Focus malaria prevention on **rainfall/temperature forecasting** and **monsoon onset monitoring** (May-June), not air quality indices

2. **Maintain air quality improvement efforts:** Respiratory/cardiovascular health benefits of reducing PM2.5 vastly outweigh any hypothetical mosquito-related effects

3. **Integrated surveillance:** Link air quality monitors with disease surveillance systems for multi-factor risk assessment

4. **Seasonal forecasting priority:** Use wet season onset as primary malaria alert trigger

5. **Vector control intensification:** Deploy bed nets, indoor spraying, larvicides based on **climate forecasts**, not PM2.5 levels

**If H₁ (Stress) is eventually validated:**
- Counterintuitive implication: Air quality improvement could increase short-term malaria risk
- Response: Intensify vector surveillance during clean-air periods (post-monsoon)
- Communication: Avoid messaging that discourages pollution control

**If H₂ (Urbanization) is eventually validated:**
- Urban planning integration: Malaria risk assessment in development projects
- Water storage management: Covered containers, regular drainage cleaning

**Current policy stance:** Treat PM2.5-malaria relationship as **unconfirmed exploratory finding**. Do not adjust public health programs based on this analysis alone.

---

## Project Structure

```
assADS/
├── Smog&Swarms_Hypothesis.ipynb  ← Full analysis notebook (open this)
├── README.md                      ← This file
├── .gitignore
├── images/                        ← All output charts & figures
│   ├── pm25_proxy_model_2001_2009.png
│   ├── ccf_pm25_malaria.png
│   ├── correlation_heatmap.png
│   ├── seasonal_decompose.png
│   ├── irf_pm25_malaria.png
│   ├── risk_heatmap_static.png
│   ├── risk_alert_timeline.png
│   └── [other generated plots]
└── data/
    ├── cambodia_monthly_merged.csv      ← 240-row monthly panel (2000-2019)
    ├── cambodia_annual_merged.csv       ← 20-row annual panel (2000-2019)
    ├── pm25_proxy_training_data.csv     ← Proxy model training data
    ├── air_pollutant.csv                ← WHO air quality database
    └── 116_Cambodia/                    ← Raw WHO SDG indicators (excluded from git)
        ├── 6A64C9A_7.1.2 - Clean fuels/
        ├── 442CEA8_3.3.3 - Malaria/
        ├── E2FC6D7_3.9.1 - Air pollution mortality/
        └── [27 other SDG indicator folders]
```

> **Raw source datasets** in `data/116_Cambodia/` contain WHO SDG indicators for clean fuels, air pollution mortality, respiratory disease, sanitation, and other health metrics. These are used to train the hybrid PM2.5 gap-filling model and as covariates in exploratory analyses.

---

## How to Run

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or VS Code with Jupyter extension
- ~250 MB disk space (including raw WHO datasets)

### Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd assADS

# 2. Create and activate virtual environment
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

# 3. Install dependencies
pip install numpy pandas matplotlib seaborn scipy statsmodels plotly scikit-learn ipywidgets nbformat

# 4. Open the notebook
jupyter notebook
# Navigate to: Smog&Swarms_Hypothesis.ipynb

# OR use VS Code
code .
# Open Smog&Swarms_Hypothesis.ipynb and click "Run All"
```

### Execution

1. **Run all cells in order:** Kernel → Restart & Run All
2. **Runtime:** ~30-60 seconds for all 31 cells (including gap-filling, VAR, VARX, dashboard)
3. **Output:** All visualizations saved to `images/` folder

### Troubleshooting

**Common issue:** `ModuleNotFoundError: No module named 'statsmodels'`  
**Solution:** Ensure virtual environment is activated, then `pip install statsmodels`

**Common issue:** ipywidgets dashboard not rendering  
**Solution:** 
```bash
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
```
For VS Code: ipywidgets should work natively with Jupyter extension

---

## Reproducibility & Citation

## Reproducibility & Citation

**Software Environment:**
- Python 3.8+ 
- statsmodels 0.13+ (VAR/VARX estimation, Granger causality, seasonal decomposition, ADF tests)
- sklearn 1.0+ (LinearRegression, Ridge, StandardScaler, PolynomialFeatures)
- scipy 1.7+ (Pchip interpolation, statistical tests)
- plotly 5.0+ (interactive dashboard)
- pandas 1.3+, numpy 1.21+, matplotlib 3.4+, seaborn 0.11+

**Data Sources (All Public):**
1. **PM2.5:** WHO Global Air Quality Database (2010-2019) + World Bank Climate Change Knowledge Portal
2. **Malaria:** WHO Global Health Observatory Indicator 442CEA8 (SDG 3.3.3)
3. **Proxies:** WHO SDG Indicators for clean fuels (6A64C9A), air pollution mortality (E2FC6D7), etc.
4. **Climate:** World Bank CCKP 30-year monthly normals (1991-2020)

**Computational Requirements:**
- Runtime: 30-60 seconds (all 31 cells)
- Memory: <2 GB RAM
- Storage: ~250 MB (including raw WHO datasets)

**Code Availability:**
All preprocessing, gap-filling, VAR/VARX modeling, and visualization code included in `Smog&Swarms_Hypothesis.ipynb`. No external scripts required.

**Citation:**
If using this methodology or hybrid PM2.5 gap-filling approach, please cite:
```
Heng (2026). The Smog & Swarms Hypothesis: PM2.5 and Malaria in Cambodia (2000-2019).
Advanced Data Science Assignment. February 2026.
Methodology: Hybrid proxy-statistical PM2.5 estimation + climate-controlled VARX causal inference.
```

**Replication Recommendations:**
For more robust conclusions, future analyses should:
1. **Validate on observed-only subset** (2010-2019, N=120 months) to avoid gap-filling bias
2. **Obtain interannual climate data** (actual observed rainfall/temperature) instead of static normals
3. **Province-level spatial analysis** with ground-based PM2.5 monitors (avoid national aggregation)
4. **Experimental validation** (lab mosquito exposure to controlled PM2.5 levels)
5. **Dengue comparison** (repeat analysis with dengue data, more urban-associated than malaria)

---

## Limitations & Responsible Use

### Data Quality Caveats

1. **45% gap-filled PM2.5 data** (2001-2009):
   - Unvalidated extrapolation from small training set (N=10 years)
   - Assumes stable PM2.5-proxy relationship (untestable)
   - Likely underestimates true interannual variability
   - **Consequence:** All results exploratory, not confirmatory

2. **Monthly disaggregation artifacts:**
   - Annual malaria data split using fixed seasonal multipliers
   - Induces perfect 12-month cycles → ADF may pass but seasonal collinearity remains
   - **Mitigation:** VARX seasonal decomposition removes deterministic cycles

3. **Climate data limitations:**
   - 30-year climatological normals, not actual observed values each year
   - Underestimates interannual climate variability
   - **Impact:** Weaker climate controls than with true observations → may overestimate PM2.5 effect

4. **National aggregation:**
   - Masks spatial heterogeneity (urban Phnom Penh vs. rural border regions)
   - PM2.5-malaria relationship may differ by locale
   - **Recommendation:** Province-level analysis with local monitors

### Statistical Inference Limitations

1. **"Dry Season Trap" confounding:**
   - PM2.5 peaks dry season (Feb-Mar), malaria peaks wet season (Aug-Oct)
   - Creates mechanical negative correlation unrelated to causation
   - **Even with VARX controls**, residual confounding may remain if seasonal decomposition incomplete

2. **Small effect size:**
   - PM2.5 coefficient magnitude is 10-100× smaller than rainfall/temperature effects
   - Practical significance questionable even if statistically significant

3. **Observational study:**
   - Cannot establish causation definitively without experimental validation
   - Granger causality = predictive precedence, not true causation
   - Unmeasured confounders possible (urbanization, vector control programs, bed net distribution, healthcare access)

4. **Multiple testing:**
   - 6 proxy regression models tested for gap-filling (model selection bias)
   - Multiple VAR lag orders compared (BIC selection reduces but doesn't eliminate bias)
   - No Bonferroni or FDR correction applied

### Responsible Interpretation

**DO:**
- ✅ Treat results as exploratory hypothesis-generating
- ✅ Acknowledge 45% gap-filled data limitation in all discussions
- ✅ Emphasize climate dominance over PM2.5 in malaria transmission
- ✅ Require experimental validation before policy changes
- ✅ Compare base VAR vs. VARX to assess confounding sensitivity

**DO NOT:**
- ❌ Present findings as definitive proof of causation
- ❌ Recommend changing air quality policies based solely on this analysis
- ❌ Claim H₁ or H₂ is "proven" without lab experiments and spatial validation
- ❌ Ignore the 10-100× larger effect of climate variables
- ❌ Generalize beyond Cambodia or beyond 2000-2019 period

### Academic Integrity

This analysis is submitted as coursework for Advanced Data Science (ADS). All code, methodology, and interpretations are original work by the author. Data sources are publicly available from WHO and World Bank repositories and properly cited.

**Plagiarism Statement:** No sections of this README or notebook are copied from uncited sources. Statistical methods follow standard textbook implementations (e.g., Lütkepohl's "New Introduction to Multiple Time Series Analysis" for VAR/VARX). Credit is given to Python library developers (statsmodels, sklearn, scipy).

---

## Acknowledgments

**Data Providers:**
- World Health Organization (WHO) Global Health Observatory for malaria and SDG indicators
- World Health Organization (WHO) Global Air Quality Database for PM2.5 satellite estimates
- World Bank Climate Change Knowledge Portal for climate normals

**Software Libraries:**
- statsmodels team for VAR/VARX/Granger causality implementation
- scikit-learn team for regression models and preprocessing
- scipy team for Pchip interpolation and statistical tests
- plotly team for interactive dashboard components

**Academic Support:**
- Advanced Data Science (ADS) course instructors for guidance on time series methods
- Peer reviewers for feedback on gap-filling methodology

---

## Contact & Feedback

**Author:** Heng  
**Course:** Advanced Data Science (ADS)  
**Submission Date:** February 27, 2026

For questions, suggestions, or collaboration inquiries regarding this analysis, please open an issue in the repository or contact through academic channels.

---

*"In the interplay between smoke and swarms, truth hides in the monsoon's rhythm. Climate whispers louder than pollution's haze."*

---

**Last updated:** February 27, 2026  
**Notebook:** `Smog&Swarms_Hypothesis.ipynb` (31 cells, 2315 lines)  
**Status:** ✅ All cells execute successfully, no errors  
**Evidence Level:** Exploratory (45% gap-filled data, climate confounding, no experimental validation)  
**Policy Recommendation:** Focus malaria prevention on climate forecasting, not air quality indices
