Hackathon
├── README.md
├── LICENSE
├── .gitignore
├── docker-compose.yml
├── .env.example
│
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI entry point
│   │   ├── config.py                  # environment variables
│   │   ├── routes/
│   │   │   ├── ingest.py
│   │   │   ├── forecast.py
│   │   │   ├── anomaly.py
│   │   │   ├── optimize.py
│   │   │   └── summary.py
│   │   ├── services/
│   │   │   ├── data_loader.py
│   │   │   ├── forecasting.py
│   │   │   ├── anomaly_detect.py
│   │   │   ├── optimization.py
│   │   │   └── utils.py
│   │   ├── models/
│   │   │   ├── forecast_model.pkl
│   │   │   ├── anomaly_model.pkl
│   │   │   └── __init__.py
│   │   ├── schemas/
│   │   │   ├── ingest_schema.py
│   │   │   ├── forecast_schema.py
│   │   │   ├── optimize_schema.py
│   │   │   └── anomaly_schema.py
│   │   └── database/
│   │       ├── bigquery.py
│   │       ├── init_db.py
│   │       └── queries.sql
│   │
│   ├── tests/
│   │   ├── test_ingest.py
│   │   ├── test_forecast.py
│   │   ├── test_anomaly.py
│   │   └── test_optimize.py
│   │
│   ├── requirements.txt
│   └── Dockerfile
│
├── data/
│   ├── raw/
│   │   └── sample_data.csv
│   ├── processed/
│   │   └── cleaned_data.parquet
│   ├── examples/
│   │   └── sample_upload.json
│   └── README.md
│
├── ml/
│   ├── notebooks/
│   │   ├── EDA.ipynb
│   │   ├── Forecasting_Dev.ipynb
│   │   └── Anomaly_Tests.ipynb
│   ├── models/
│   │   ├── train_forecast.py
│   │   ├── train_anomaly.py
│   │   └── saved_models/
│   │       ├── forecast.pkl
│   │       ├── anomaly.pkl
│   │       └── scaler.pkl
│   ├── utils/
│   │   ├── preprocessing.py
│   │   ├── feature_engineering.py
│   │   └── eval_metrics.py
│   └── requirements.txt
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── index.js
│   │   │   ├── forecast.js
│   │   │   ├── alerts.js
│   │   │   ├── optimize.js
│   │   │   └── upload.js
│   │   ├── components/
│   │   │   ├── Layout.jsx
│   │   │   ├── ChartCard.jsx
│   │   │   ├── AlertsTable.jsx
│   │   │   └── Sidebar.jsx
│   │   ├── hooks/
│   │   │   └── useApi.js
│   │   ├── styles/
│   │   │   └── globals.css
│   │   └── utils/
│   │       └── formatters.js
│   ├── package.json
│   ├── next.config.js
│   └── Dockerfile
│
└── deployment/
    ├── kubernetes/
    │   ├── backend-deployment.yaml
    │   ├── backend-service.yaml
    │   ├── frontend-deployment.yaml
    │   └── frontend-service.yaml
    ├── cloudbuild.yaml
    ├── terraform/
    └── README.md
