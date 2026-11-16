# Veritas AI International High School Hackathon 2026 — Submission Pack


---

## Part 1: Submission Form Answers (Copy-Paste Ready)

| Form Field | Ready Answer |
| :--- | :--- |
| **Project title** | **SaansCheck**: Next-Day Delhi AQI, Spike & Safe-Window Forecaster |
| **Building as** | Team |
| **Team name** | **SaansCheck** |
| **Team members** | **Yatharth Dadu**, **Debyash Mohanty**, **Advay Kongorpalli** *(enter everyone's exact name, email, and school on the form)* |
| **Track** | AI + Climate |
| **Starting point** | 4.3.2 Next-day air quality forecast |
| **One-sentence pitch** | SaansCheck helps a Delhi school coach and students with asthma plan tomorrow's outdoor schedule tonight by forecasting expected Indian AQI, catching sudden smog spikes, and finding the safest 2-hour outdoor window. |
| **Code link** | `https://github.com/YatharthDadu/SaansCheck` *(Colab direct launch: `https://colab.research.google.com/github/YatharthDadu/SaansCheck/blob/main/Delhi_AQI_NextDay_Forecast.ipynb`)* |
| **Write-up link** | `https://github.com/YatharthDadu/SaansCheck#readme` *(or paste Part 2 below into a Google Doc set to "Anyone with the link can view" / export PDF)* |
| **Demo video link** | *(Required by the official checklist — paste unlisted YouTube or Drive link using the 2-minute script in Part 3 below)* |
| **What we added** | **(1) Combined two real Delhi school users in one tool:** a School Coach/Principal mode using India CPCB/GRAP thresholds and an Asthma Student mode forecasting tomorrow's safest 2-hour outdoor window from an hourly diurnal model. **(2) Asymmetric spike safety layer:** trained a 72nd-percentile quantile model to catch sudden overnight spikes and Poor/Very Poor days that the "tomorrow = today" persistence baseline misses completely. **(3) Real ground-sensor bias audit:** compared Open-Meteo's modeled CAMS estimates against 61 days of physical Delhi CPCB monitors (Oct–Nov 2024) and quantified the under-prediction gap. |
| **Datasets, models and AI tools used** | Open-Meteo Air Quality API (12 global hourly pollutants, CC BY 4.0); Open-Meteo Historical Weather API (7 hourly meteorological variables including boundary layer height, CC BY 4.0); CPCB ground monitors as archived by xKDR (Oct–Nov 2024 audit); scikit-learn (`Ridge`, `HistGradientBoostingRegressor`, `DecisionTreeRegressor`); pandas, NumPy, Matplotlib, Gradio; AI coding assistant (used to structure data fetching, feature engineering, and debugging; all code, splits, and results verified by the team). |
| **New to AI?** | *(Select the option that fits your team accurately)* |

---

## Part 2: One-Page Project Write-Up (Official Template)

# SaansCheck — Next-Day Delhi AQI & Safe-Window Forecaster
**Track:** AI + Climate (Starting point: 4.3.2 Next-day air quality forecast)  
**Team:** Yatharth Dadu, Debyash Mohanty, Advay Kongorpalli  

> **Prototype disclaimer:** SaansCheck is a student prototype, not medical advice and not a substitute for official CPCB / government health advisories. It forecasts modeled Copernicus CAMS estimates (not ground-sensor readings), which ran about 1.7x lower than Delhi CPCB monitors in our check, so when in doubt treat the air as one AQI band worse than shown.

### The problem
Every October and November, hazardous smog forces Delhi schools to scramble at the last minute over whether to cancel outdoor sports and morning assemblies, while students with asthma struggle to know which hours of the day are safe to step outside. Having lived in Delhi for 10 years, we experienced these sudden winter pollution spikes firsthand. Standard AQI apps report today's reading or assume tomorrow will look like today—reacting one day too late when calm winds and low night inversions trap toxic smoke overnight.

### What we built
We built **SaansCheck**, a unified next-day air quality planner for Delhi (`28.6139° N, 77.2090° E`) powered by a single multi-year dataset stitched live to **Indian Standard Time (IST)** and two connected `scikit-learn` models (`RUN_APP = False` default in Colab, with an optional Gradio interface featuring a **"Who Are You Planning For?"** toggle). All statistical metrics stay in the background while end-users see only familiar **Indian AQI** and a plain-language plan of action:
- **School Principal / Teacher / Coach Mode:** Forecasts tomorrow's Indian AQI and gives an instant schedule decision for morning assembly, PE/sports practice, and the cleanest afternoon outdoor slot.
- **Student / Parent (with Asthma) Mode:** Forecasts tomorrow's 24-hour curve to flag which peak morning smog hours to avoid, when to wear an N95 mask, and the **safest 2-hour daytime outdoor window**.

### What we added
Beyond the basic 4.3.2 next-day regression baseline, we added four original extensions:
1. **Live IST-Synced Forecasting + Plain-Language AQI UX:** Uses `time` + IST (`UTC+5:30`) to stitch historical Open-Meteo archives with real-time forecasts so teachers and students always get tomorrow's live Indian date and AQI action plan without technical jargon.
2. **Unified Daily + Hourly Safe-Window Engine:** Paired our daily model with a 24-hour diurnal predictor so users get both an indoor/outdoor decision and the cleanest 2-hour window if outdoor activity is necessary.
3. **Asymmetric Safety Layer (`Q72`) for Sudden Spikes:** In Delhi, missing a hazardous smog jump exposes students to toxic air, whereas an extra indoor alert is a low-cost precaution. In quantile regression, the pinball loss ratio $q/(1-q) = 0.72/0.28 \approx 2.57:1$ penalizes missed toxic spikes ~2.6× more heavily than false alarms. We fixed q = 0.72 as a safety-leaning setting; we did not run a search or cross-validation over q, and we report its test results as-is.
4. **Modeled CAMS vs. Real CPCB Ground Audit:** Because Open-Meteo air quality data is based on Copernicus CAMS atmospheric model estimates rather than physical ground monitors, we cross-checked 61 days in Oct–Nov 2024 against real Delhi CPCB stations (CPCB data as archived by xKDR, requiring no API key or token).

### How it works
- **Data:** **37,584 hourly records** (`2022-08-05` to `2026-11-17` IST) from Open-Meteo Air Quality (all 12 global variables: `pm2_5`, `pm10`, `carbon_monoxide`, `nitrogen_dioxide`, `sulphur_dioxide`, `ozone`, `aerosol_optical_depth`, `dust`, `uv_index`, `uv_index_clear_sky`, plus `carbon_dioxide` and `methane` available from Oct 26, 2024 onward; the 7 Europe-only pollen/ammonia variables were excluded) stitched with 7 Open-Meteo weather variables (`temperature_2m`, `relative_humidity_2m`, `precipitation`, `surface_pressure`, `wind_speed_10m`, `wind_direction_10m`, and `boundary_layer_height` from ERA5 archive + live `/v1/forecast` for the recent archive-lag window and tomorrow IST). After 7-day rolling features and lags, we obtained **1,559 daily rows**.
- **Strict Chronological Split (Zero Leakage):** Trained on **1,146 days** (`2022-08-12` to `2025-09-30`) and evaluated strictly on **411 completed unseen future days** (`2025-10-01` to `2026-11-15`), keeping `today_ist` (`2026-11-16`) and `tomorrow_ist` (`2026-11-17`) strictly for live inference.
- **Features & Models:** Engineered evening vs. daily $\text{PM}_{2.5}$ momentum, fine-to-coarse smoke ratio ($\text{PM}_{2.5}/\text{PM}_{10}$), northwest wind share ($270^\circ\text{–}360^\circ$), diurnal temperature range, stagnation index, and tomorrow's meteorological features (`wind_speed_10m`, `boundary_layer_height`, `precipitation`). Note: in training and testing these are realised ERA5 weather for day t+1 standing in for a weather forecast (only the live demo uses a true forecast), so test scores are slightly optimistic; see the weather-noise check below. We compared **Persistence** ($\hat{y}_{t+1}=y_t$), **Ridge Regression**, an unregularized **Decision Tree** (rejected for overfitting), a **Day-$t$-only HistGBR** ablation, and our final **HistGradientBoostingRegressor (HistGBR)** + **Q72 Safety Bound**.

### Results
Evaluated on the **411 completed unseen test days** (`2025-10-01` to `2026-11-15`):

| Model | Overall MAE ($\mu\text{g/m}^3$) | Smog Season MAE (Oct–Feb) | High-Change Jump MAE ($\lvert\Delta\rvert \ge 20$) | Poor Days Recall ($>90\,\mu\text{g/m}^3$) | Sudden Spike Recall ($+20\,\mu\text{g/m}^3$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline: Persistence** (*Tomorrow = Today*) | 17.66 | 18.81 | 37.66 | 76.9% (36 FP) | 0.0% (0/64; always 0 by construction, since persistence never forecasts a rise) |
| **Rejected: DecisionTree** | 19.78 | 19.07 | 29.66 | 60.3% (32 FP) | 21.9% (31 FP) |
| **Model 1: Ridge Regression** | 13.70 | 12.78 | 22.50 | 79.5% (26 FP) | 29.7% (12 FP) |
| **Ablation: HistGBR (Day-$t$ only)** | 13.80 | 14.62 | 22.81 | 75.6% (28 FP) | 15.6% (9 FP) |
| **Model 2: HistGBR (Full)** | **12.83** | **12.33** | **20.65** | 78.8% (27 FP) | 21.9% (6 FP) |
| **Model 3: Q72 Safety alert layer** | 13.64 (own forecast; alerts use Q72, point forecast is HistGBR) | — | — | **88.5%** (38 FP) | 46.9% (30/64, 45 FP; alert rule is ≥ +15 for Q72 vs ≥ +20 for the others) |

- **Key Forecast Gain:** `HistGBR` cuts overall MAE by **27.3%**, smog-season MAE by **34.5%**, and high-change jump-day MAE by **45.2%** compared to Persistence (`R² = 0.781` vs. `0.590`). On the US EPA threshold (`>35 µg/m³`, met on 394/411 test days), `HistGBR` achieves 99.7% recall (1 missed day), but this threshold is nearly trivial in Delhi (an "always warn" rule gets 100% recall at 95.9% precision), so we focus on the CPCB thresholds. The "high-change jump" MAE column picks days using tomorrow's answer (persistence error is ≥ 20 there by definition), so treat it as a secondary diagnostic, not a headline.
- **Safe-Window Impact (Hourly Model):** Hourly `HistGBR` reduces 24-hour MAE from `26.67` (24h persistence) to **`18.56 µg/m³`**. Scheduling outdoor activity during the model's recommended 2-hour window instead of morning assembly (`07:00–09:00 IST`) cuts average $\text{PM}_{2.5}$ exposure by **37.1%** (`96.75` → `60.87 µg/m³`) across all test days and **39.2%** (`149.97` → `91.21 µg/m³`) across the 125 smog days (`>90 µg/m³`), landing within $10\,\mu\text{g/m}^3$ of the true cleanest window on **81.8%** of days.
- **Top Drivers & Real Sensor Gap:** Permutation importance confirms evening $\text{PM}_{2.5}$ (`+11.14 µg/m³`), daily mean $\text{PM}_{2.5}$, next-day wind speed, and boundary layer height drive next-day air quality. Cross-checking Oct–Nov 2024 against physical **CPCB ground monitors** ($n=61$ days) revealed moderate correlation ($r = 0.658$) but a large scale gap: CAMS averaged `91.9 µg/m³` (max `176.2`) while real Delhi CPCB stations averaged **`178.6 µg/m³`** (max **`582.8`**, median ratio **`1.66×`**).

**Hazardous days (CPCB "Very Poor", `>120 µg/m³`; 71 of 411 test days).** Honest result: the ML model improves average error much more than it improves detection of the worst days.

| Model | Precision | Recall | F1 | False-positive rate | Test RMSE / R² |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Persistence | 0.676 | 0.676 | 0.676 | 6.8% | 24.26 / 0.590 |
| Ridge | 0.794 | 0.704 | **0.746** | 3.8% | 18.79 / 0.754 |
| Decision Tree | 0.735 | 0.507 | 0.600 | 3.8% | 27.13 / 0.487 |
| HistGBR (selected) | 0.780 | 0.648 | 0.708 | 3.8% | **17.74 / 0.781** |
| HistGBR + Q72 alert | 0.754 | 0.690 | 0.721 | 4.7% | 18.23 / 0.768 (Q72 alone) |

- At `>120`, HistGBR's recall (64.8%) is slightly *below* persistence (67.6%); it wins on precision and false alarms, and Ridge has the best F1. Q72 recovers some recall (69.0%) for 3 extra false alarms. Its bigger gain is on the `>90` "Poor" threshold (88.5% recall) at the cost of 11 more false alarms than HistGBR (38 vs 27).
- **Overfitting check (train MAE → test MAE):** Decision Tree 0.00 → 19.78 (memorises training data, worse than persistence), Ridge 9.91 → 13.70, HistGBR 4.61 → 12.83. HistGBR is regularised (`max_depth=4`, `l2_regularization=1.0`, fixed 250 iterations; no early stopping, which scikit-learn disables for small data), and still shows a train/test gap.
- **Weather-forecast realism:** adding random noise to tomorrow's weather features at test time (0.25 / 0.5 / 1.0 training standard deviations) moves HistGBR MAE from 12.83 to 13.16 / 13.52 / 14.75, and removing them entirely gives 13.80; all still beat persistence (17.66).
- **How q=0.72 was set:** q = 0.72 is a fixed setting chosen for its pinball-loss ratio q/(1-q) = 0.72/0.28 ≈ 2.57:1, which penalises under-predicting a hazardous day about 2.6x more than a false alarm. No search or cross-validation over q was run, so read it as a safety-leaning heuristic, not an optimised value. On the test set it delivered 88.5% Poor recall and 69.0% Very Poor recall. It is not a guaranteed upper bound: 64% of test days fell below Q72 (nominal 72%), and Q72 sat below the HistGBR forecast on 17.5% of days.
- **Safe-window baseline:** a fixed 13:00-15:00 rule averages 62.35 µg/m³ vs 60.87 for the hourly model (within 10 µg/m³ of the best window on 79.6% vs 81.8% of days). Most of the 37% saving over the morning assembly therefore comes from Delhi's daily cycle; the hourly model adds about 1.5 µg/m³ on top.

### Where it fails
1. **Summer / Pre-Monsoon Dust Storms:** Our largest test errors occurred on `2026-06-11` (`177.6` actual vs. `101.0` predicted, wind `12.3 km/h`) and `2026-05-21` (`138.9` actual vs. `71.2` predicted, wind `13.7 km/h`). In winter, strong winds disperse smog; in May–June, strong winds *import* regional desert dust, breaking the model's learned wind-clearing rule. The data supports this: on these days (and `2026-04-10`, `2026-07-11/12`) CAMS dust is about 1,300 vs a test median of 119, and PM2.5/PM10 is only 0.14-0.23. About a third of the test period's "Very Poor" days (24 of 71) fall in May-July.
2. **Sudden Unobserved Emission Surges & CAMS Smoothing:** Winter outlier `2025-12-12` (`161.1` actual vs. `104.8` predicted under `3.6 km/h` calm winds and a `219 m` boundary layer) shows that local weather cannot anticipate abrupt upstream emission bursts. CAMS also smooths extreme street-level spikes: it shows almost no Diwali 2025 spike (actuals of 85-109 µg/m³ on `2025-10-20` to `2025-10-22`), so we could not evaluate firework or crop-fire shocks.
3. **CAMS reads low vs ground sensors:** our forecasts and AQI bands are on the CAMS scale, which averaged about 1.7x lower than CPCB monitors in Oct-Nov 2024. A "Moderate" forecast may be "Poor" at street level, so the action plan is a planning aid only; we have not calibrated it to CPCB stations (see next steps).

### What we'd do next
1. **Calibrate Directly on Real CPCB Stations:** Train a station-level bias-correction layer using multi-year CPCB/xKDR data so school alerts reflect true hyperlocal ground concentrations rather than smoothed CAMS grid estimates.
2. **Add Regional Dust & Fire Signals:** Incorporate upwind fire-radiative power (NASA FIRMS) and separate a pre-monsoon dust regime (`Apr–Jun`) from the post-monsoon combustion regime (`Oct–Jan`).

### Sources
- **Open-Meteo Air Quality API (CAMS, CC BY 4.0):** [https://open-meteo.com/en/docs/air-quality-api](https://open-meteo.com/en/docs/air-quality-api)
- **Open-Meteo Historical Weather API (ERA5, CC BY 4.0):** [https://open-meteo.com/en/docs/historical-weather-api](https://open-meteo.com/en/docs/historical-weather-api)
- **CPCB data as archived by xKDR:** [https://github.com/xKDR/Air-Quality-Database](https://github.com/xKDR/Air-Quality-Database)
- **Tools & Libraries:** Python, pandas, NumPy, scikit-learn ([https://scikit-learn.org](https://scikit-learn.org)), Matplotlib, Gradio ([https://www.gradio.app](https://www.gradio.app))
- **AI Tools Disclosure:** We used an AI coding assistant to help write data-fetching scripts, structure feature engineering, and debug code; our team designed the unified school/asthma framing, selected variables, ran and verified all Colab/pipeline experiments, and wrote the analysis.

---

## Part 3: Optional 2-Minute Demo Video Script & Submission Checklist

### 2-Minute Voiceover Script (Screen Recording of Colab / Gradio)
- **0:00–0:15 (The Problem):** *"Having lived in Delhi for 10 years, we've seen how November smog shuts down schools overnight and leaves students with asthma guessing when it's safe to go outside. Standard apps only tell you today's AQI—or assume tomorrow equals today."*
- **0:15–1:15 (The Demo — Show Cell 8 / Gradio & Charts):** *"Here is SaansCheck. Using live Indian Standard Time data for Delhi, a school principal opens the app and immediately sees tomorrow's forecast—for instance November 17 in India—with plain-English Indian AQI and an exact schedule recommendation. Or look at a heavy smog jump like November 8: our model predicts AQI ~307 (Very Poor) and tells the school to move assembly and sports indoors, while suggesting 1:00 PM to 3:00 PM as the cleanest outdoor window. Toggle to Student with Asthma mode on the exact same engine, and it tells you which peak morning hours to avoid and the safest 2-hour afternoon slot—cutting smog-day exposure by 39%."*
- **1:15–1:45 (How It Works & Baseline Comparison):** *"We trained HistGradientBoosting models strictly on 2022–2025 data and tested on 411 completed unseen days through November 15, 2026. Compared to Persistence, our model cuts overall MAE by 27%, cuts high-jump day error by 45%, and with a Q72 safety buffer Poor-day recall rises from 79% to 88.5%, at the cost of more false alarms. We also checked 61 days against real CPCB ground sensors and found CAMS underestimates peak ground smog by ~1.7x."*
- **1:45–2:00 (Limitations & Next Steps):** *"Our biggest errors happen during May–June dust storms when high winds actually bring pollution in rather than clearing it. Next, we'd calibrate directly to local CPCB stations and add upwind NASA fire data. This is a prototype, not medical advice."*

### Pre-Submit Checklist (Do Before 11:59 PM ET!)
1. Upload `Code/Delhi_AQI_NextDay_Forecast.ipynb` to **Google Colab** (`colab.research.google.com`).
2. Optionally set `RUN_APP = True` briefly to test/record the Gradio UI, then set `RUN_APP = False` and click **Runtime → Run all** so every cell runs cleanly with outputs visible.
3. Click **Share → Anyone with the link (Viewer)** and test the Colab link + Google Doc write-up link in an **Incognito/Private browser window**!
4. Fill in your chosen **SaansCheck** and all three names/emails/schools on the submission form, and submit at least 15–20 minutes early!
