# Property Price-per-Unit-Area Estimator (Prototype)

## What this is

A prototype machine learning tool that gives a **preliminary, data-driven
estimate** of a property's price per unit area, based on a handful of
basic facts about the property and its location. It is meant to support
an agent's **initial review** of a listing — a quick sanity check before
they invest time in a full market analysis.

**This tool does not produce a formal appraisal.** A licensed appraisal
considers factors this prototype has no access to — interior condition,
renovations, comparable sales chosen by a human expert, legal and zoning
issues, and local market judgment. The number below is a statistical
estimate based on patterns in historical sales data, nothing more.

## Data source

This prototype uses the public **UCI Real Estate Valuation** dataset:

> Yeh, I. (2018). Real Estate Valuation [Dataset]. UCI Machine Learning
> Repository. https://doi.org/10.24432/C5J30W

The dataset is licensed under **CC BY 4.0**. It contains 414 real
transactions of residential units in Sindian District, New Taipei City,
Taiwan, collected between 2012 and 2013.

The pipeline (`src/train_model.py`) loads the data with the official
`ucimlrepo` loader when internet access to UCI is available. A local
copy (`data/Real estate valuation data set.xlsx`, the same public
dataset) is bundled as a fallback so the project remains fully
reproducible in offline or restricted-network environments.

## The six inputs the model uses

| Column | Meaning | Decision |
|---|---|---|
| `No` | Row number / record ID | **Excluded.** It is only an identifier assigned when the data was compiled; it carries no real information about the property and would just let the model memorize row order, which means nothing for a new listing. |
| `X1 transaction date` | Sale date, as a decimal year (e.g. 2013.25) | Kept as-is. It's already numeric and captures short-term market trend over the ~1 year the data spans. |
| `X2 house age` | Age of the property in years | Kept as-is. Directly relevant to value. |
| `X3 distance to the nearest MRT station` | Distance in meters | Kept as-is. Right-skewed (a few properties are far outliers), which the tree-based model handles natively without transformation. |
| `X4 number of convenience stores` | Count of nearby convenience stores | Kept as-is. A simple, meaningful proxy for neighborhood walkability/amenities. |
| `X5 latitude`, `X6 longitude` | Geographic coordinates | Kept as-is. Location is one of the strongest drivers of real estate price. |

All six features are numeric with **no missing values**, so no imputation
was needed. Each numeric feature was standardized (mean 0, std 1) before
modeling — this doesn't matter much for the tree-based model but keeps
the linear baseline well-behaved and makes it easy to swap models later.

## Modeling approach

1. **Train/test split first.** The data was split 80/20 *before* any
   modeling decisions (feature choices, model type, tuning) were made
   using the test portion. This matters — see "Why not just test on
   the training data?" below.
2. **Baseline model:** ordinary linear regression, for reference.
3. **Primary model:** a Random Forest regressor. It captured the
   non-linear relationship between MRT distance and price noticeably
   better than the linear baseline.
4. **Cross-validation:** in addition to the single held-out test set,
   5-fold cross-validation was run on the training data only, to check
   that performance is stable and not a fluke of one particular split.

## How well does it work?

On the **held-out test set** (the 20% of properties the model never saw
during training):

| Model | Mean Absolute Error | Mean Absolute % Error | R^2 |
|---|---|---|---|
| Linear Regression (baseline) | ~5.3 | ~17% | ~0.68 |
| **Random Forest (primary)** | **~3.7** | **~12%** | **~0.82** |

*(Exact numbers are written to `outputs/model_metrics.txt` each time the
script is run and can shift slightly between runs.)*

**In plain terms:** on properties the model hadn't seen, its estimate of
price per unit area was off by about 3.7 units on average (roughly 12%
of a typical property's value), and the model explains about 82% of the
variation in prices across properties. That's a reasonable starting
point for a quick screening tool, but it also means individual estimates
can be meaningfully wrong — hence the "not an appraisal" caveat.

### Why not just test on the training data?

If we evaluated the model on the same rows it was trained on, it could
score very well simply by **memorizing** those specific transactions —
including their noise and quirks — rather than learning patterns that
generalize to new properties. That would make the model look far more
accurate than it really is. The only honest way to estimate how the
model will perform on a brand-new listing is to test it on data it
never saw during training, which is what the reported metrics above do.

## Why "estimate," not "appraisal"

- The model only knows six numeric facts about a property. A real
  appraisal considers interior condition, layout, renovations, view,
  noise, school zones, legal encumbrances, and the appraiser's judgment
  of the most comparable recent sales — none of which this model sees.
- The output is a statistical average behavior learned from historical
  data, with a measured error margin (see above) — it is a starting
  point for the agent's own analysis, not a substitute for it.

## Why this may not generalize to a different market or time period

This dataset covers one district in New Taipei City over about one
year (2012–2013). A model trained on it has learned the relationships
between price, age, MRT distance, and amenities **as they existed in
that specific market and period**. Those relationships change over
time (interest rates, local development, overall price levels) and
across places (a different city's price drivers, transit systems, and
neighborhood layouts won't match Sindian District's). Applying this
exact trained model to a different city, country, or time period would
likely produce unreliable estimates — the tool should be retrained on
local, current data before being used for another market.

## Example predictions

Three example properties (not in the dataset) were used to demonstrate
the model, written to `outputs/new_property_predictions.csv`:

| Property | Age | Distance to MRT (m) | Convenience stores | Estimated price per unit area |
|---|---|---|---|---|
| A: newer, close to MRT, several stores nearby | 5 | 200 | 8 | ~51.2 |
| B: older, far from MRT, few stores nearby | 30 | 1500 | 1 | ~22.3 |
| C: mid-age, moderate distance, average amenities | 15 | 600 | 4 | ~40.3 |

Each of these is a **preliminary estimate to guide an agent's initial
review** — not a certified value.

## Project structure

```
.
├── README.md                     <- this file
├── requirements.txt               <- Python dependencies
├── data/
│   └── Real estate valuation data set.xlsx   <- bundled fallback copy of the UCI dataset
├── src/
│   └── train_model.py            <- full pipeline: load, explore, split, preprocess, train, evaluate, predict
└── outputs/                      <- generated each run
    ├── eda_summary.txt
    ├── model_metrics.txt
    ├── new_property_predictions.csv
    └── model_pipeline.joblib
```

## How to reproduce

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/train_model.py
```

This will print exploration notes, evaluation metrics, and example
predictions to the console, and write the same information to files
under `outputs/`. Results (loading path, headline metric numbers) may
vary by a small amount between the UCI-loader path and the bundled
local file only in nonessential precision; the modeling code and random
seed are otherwise identical.

## Limitations recap for the customer

- Small dataset (414 transactions) from a single district and ~1-year
  window.
- No information on interior condition, renovation, or unit-specific
  features beyond age.
- Estimates carry a measured error margin (~12% MAPE on held-out data)
  and should always be paired with an agent's judgment.
- Not valid for other cities, countries, or materially different time
  periods without retraining on relevant local data.
