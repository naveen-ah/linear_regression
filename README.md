# End-to-End Linear Regression for Real-Time Predictions

This project builds a complete linear-regression-based machine learning workflow that includes:

- Exploratory Data Analysis (EDA)
- Feature engineering
- Feature selection
- Model selection/tuning across linear models
- Model persistence
- Real-time prediction API using FastAPI

The implementation uses the California Housing dataset from scikit-learn.

## Project structure

```text
.
├── README.md
├── requirements.txt
├── models/                 # generated model artifacts
├── reports/                # generated EDA artifacts
└── src/
    ├── train.py            # full training pipeline
    └── predict_api.py      # real-time inference API
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train the model (EDA + feature engineering + feature selection)

```bash
python src/train.py
```

Training will generate:

- `reports/eda_summary.json`
- `reports/correlation_heatmap.png`
- `reports/target_distribution.png`
- `models/linear_regression_pipeline.joblib`
- `models/metrics.json`

## What the pipeline does

1. **EDA**
   - Dataset summary stats
   - Missing value report
   - Correlation analysis with target
   - Target distribution plot

2. **Feature engineering**
   - `rooms_per_person`
   - `bedrooms_per_room`
   - `population_per_household`
   - `distance_to_coast_proxy`

3. **Feature preprocessing**
   - Median imputation
   - Standard scaling
   - Polynomial interaction features

4. **Feature selection**
   - `SelectFromModel` with `LassoCV`

5. **Linear model selection**
   - `LinearRegression`
   - `Ridge`
   - `ElasticNet`
   - Best model selected via `GridSearchCV` on RMSE

## Run real-time prediction API

Make sure the model is trained first.

```bash
uvicorn src.predict_api:app --reload --host 0.0.0.0 --port 8000
```

### Health check

```bash
curl http://localhost:8000/health
```

### Predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.984,
    "AveBedrms": 1.023,
    "Population": 322.0,
    "AveOccup": 2.555,
    "Latitude": 37.88,
    "Longitude": -122.23
  }'
```

Response:

```json
{"prediction": 4.1}
```

> Prediction value is in 100,000 USD units (dataset convention).
