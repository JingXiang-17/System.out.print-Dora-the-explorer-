# backend.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import joblib
import os

app = FastAPI(title="Flight Delay & Optimization API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# CONFIGURATION
# -----------------------------
TARGET_COLUMNS = ['CARRIER_DELAY', 'SECURITY_DELAY', 'WEATHER_DELAY']
MODEL_FILENAME = 'flight_delay_predictor.joblib'
TAIL_NUMBER_COL = 'TAIL_NUMBER'
CATEGORICAL_COLS_FOR_XGB = [
    'ORIGIN_AIRPORT_CODE', 'DEST_AIRPORT_CODE', 'MARKETING_AIRLINE', 
    'ORIGIN_STATE', 'DEST_STATE', 'DEPARTURE_BLOCK', 'ARRIVAL_BLOCK'
]

# -----------------------------
# SUGGESTIONS DATABASE
# -----------------------------
carrier_suggestions = [
    "Check airline announcements for potential delays.",
    "Choose flights from carriers with high on-time performance.",
    "Arrive early to avoid long check-in queues.",
    "Avoid peak travel periods when airlines are heavily congested.",
    "Use airline mobile apps for real-time updates.",
    "Avoid tight layovers; keep at least 2 hours buffer.",
    "Check baggage rules to avoid extra processing delays.",
    "Complete online check-in early.",
    "If airline delay is confirmed, ask about rebooking options.",
    "Morning flights are generally more punctual—consider choosing them."
]

weather_suggestions = [
    "Check weather forecasts for departure and destination.",
    "Avoid flights during seasons with frequent storms or fog.",
    "Keep extra travel buffer time during unstable weather.",
    "Bring portable chargers for long waiting hours.",
    "Track real-time gate updates at the airport.",
    "Use flight tracking apps like FlightAware.",
    "Plan your transportation early to avoid weather-caused traffic delays.",
    "Prefer major hub airports during bad weather seasons.",
    "Choose morning flights for more stable weather conditions.",
    "Request free rebooking if extreme weather occurs."
]

security_suggestions = [
    "Arrive 2–3 hours early to account for security delay.",
    "Avoid carrying liquids or metal items unnecessarily.",
    "Use automated e-gates if available.",
    "Avoid peak periods such as 7–9am and 5–7pm.",
    "Prepare electronics and documents in advance.",
    "Check if your airport provides Fast Track security lanes.",
    "Avoid clothing with heavy metal accessories.",
    "Follow airport social media for real-time congestion updates.",
    "Travel with only carry-on bags to skip baggage queues.",
    "Book earlier flights during festive / peak seasons."
]

# -----------------------------
# GLOBAL STORAGE
# -----------------------------
uploaded_df = pd.DataFrame()
predictions_df = pd.DataFrame()
anomalies_df = pd.DataFrame()

# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------
def load_feature_names(filename='feature_names.txt'):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Feature names file '{filename}' not found.")
    with open(filename, 'r') as f:
        return [line.strip() for line in f]

def transform_data_for_inference(df_new, trained_feature_names):
    # Rename columns
    column_mapping = {
        'MKT_UNIQUE_CARRIER': 'MARKETING_AIRLINE',
        'OP_UNIQUE_CARRIER': 'OPERATING_AIRLINE', 
        'ORIGIN': 'ORIGIN_AIRPORT_CODE', 'ORIGIN_STATE_ABR': 'ORIGIN_STATE',
        'DEST': 'DEST_AIRPORT_CODE', 'DEST_STATE_ABR': 'DEST_STATE',
        'DEP_TIME_BLK': 'DEPARTURE_BLOCK', 'ARR_TIME_BLK': 'ARRIVAL_BLOCK'
    }
    df_new = df_new.rename(columns=column_mapping)
    
    # Fill missing numeric values
    for col in df_new.select_dtypes(include=[np.number]).columns:
        df_new[col] = df_new[col].fillna(df_new[col].median())
    
    # Feature engineering for time
    if 'CRS_DEP_TIME' in df_new.columns:
        df_new['DEP_HOUR'] = (df_new['CRS_DEP_TIME'] // 100).astype(np.int8)
        df_new['DEP_MINUTE'] = (df_new['CRS_DEP_TIME'] % 100).astype(np.int8)
        df_new['DEP_HOUR_SIN'] = np.sin(2*np.pi*df_new['DEP_HOUR']/24).astype(np.float32)
        df_new['DEP_HOUR_COS'] = np.cos(2*np.pi*df_new['DEP_HOUR']/24).astype(np.float32)
        df_new['DEP_MIN_SIN'] = np.sin(2*np.pi*df_new['DEP_MINUTE']/60).astype(np.float32)
        df_new['DEP_MIN_COS'] = np.cos(2*np.pi*df_new['DEP_MINUTE']/60).astype(np.float32)
    for col in ['MONTH','DAY_OF_WEEK']:
        if col in df_new.columns:
            div = 12 if col=='MONTH' else 7
            df_new[f'{col}_SIN'] = np.sin(2*np.pi*df_new[col]/div).astype(np.float32)
            df_new[f'{col}_COS'] = np.cos(2*np.pi*df_new[col]/div).astype(np.float32)
    
    # Convert categorical columns
    for col in CATEGORICAL_COLS_FOR_XGB:
        if col in df_new.columns:
            df_new[col] = df_new[col].astype('category')
        else:
            df_new[col] = pd.Series(np.nan, index=df_new.index, dtype='category')
    
    # One-hot encode OPERATING_AIRLINE if exists
    if 'OPERATING_AIRLINE' in df_new.columns:
        df_new = pd.get_dummies(df_new, columns=['OPERATING_AIRLINE'], prefix='AIRLINE', dtype='int8')
    
    # Align with trained feature names
    final_df = pd.DataFrame(index=df_new.index)
    for feature in trained_feature_names:
        final_df[feature] = df_new[feature] if feature in df_new.columns else 0
    return final_df

def risk_label(value):
    av = abs(value)
    if av < 15: return 'LOW'
    if av <= 60: return 'MEDIUM'
    return 'HIGH'

def detect_anomalies(pred_df):
    df = pd.DataFrame()
    if TAIL_NUMBER_COL in pred_df.columns:
        df[TAIL_NUMBER_COL] = pred_df[TAIL_NUMBER_COL]
    for col in TARGET_COLUMNS:
        df[f"{col}_RISK"] = pred_df[col].apply(risk_label)
    return df

def optimize_suggestions(row):
    suggestions = []
    if row['CARRIER_DELAY'] > 15:
        suggestions.extend(carrier_suggestions[:5])
    elif row['CARRIER_DELAY'] > 5:
        suggestions.append(carrier_suggestions[0])
    if row['WEATHER_DELAY'] > 10:
        suggestions.extend(weather_suggestions[:5])
    elif row['WEATHER_DELAY'] > 5:
        suggestions.append(weather_suggestions[0])
    if row['SECURITY_DELAY'] > 5:
        suggestions.extend(security_suggestions[:5])
    elif row['SECURITY_DELAY'] > 2:
        suggestions.append(security_suggestions[0])
    if row['Total_Predicted_Delay'] > 20:
        suggestions.append("High total predicted delay; consider alternative flights or adding buffer time.")
    return suggestions

# -----------------------------
# Pydantic Model for JSON input
# -----------------------------
class FlightData(BaseModel):
    TAIL_NUMBER: Optional[str]
    ORIGIN: str
    DEST: str
    CRS_DEP_TIME: Optional[int]
    MONTH: Optional[int]
    DAY_OF_WEEK: Optional[int]
    DEP_TIME_BLK: Optional[str]
    ARR_TIME_BLK: Optional[str]
    MKT_UNIQUE_CARRIER: Optional[str]
    OP_UNIQUE_CARRIER: Optional[str]
    # 可根据需要加更多字段

# -----------------------------
# FASTAPI ENDPOINTS
# -----------------------------
# CSV Upload (existing)
# JSON input (new) -> this is what Flutter should call
@app.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...)):
    global uploaded_df, predictions_df, anomalies_df
    try:
        uploaded_df = pd.read_csv(file.file)
        feature_names = load_feature_names()
        X_inf = transform_data_for_inference(uploaded_df, feature_names)
        model = joblib.load(MODEL_FILENAME)
        y_pred = model.predict(X_inf)
        predictions_df = pd.DataFrame(y_pred, columns=TARGET_COLUMNS)
        predictions_df['Total_Predicted_Delay'] = predictions_df.sum(axis=1)
        if TAIL_NUMBER_COL in uploaded_df.columns:
            predictions_df.insert(0, TAIL_NUMBER_COL, uploaded_df[TAIL_NUMBER_COL])
        anomalies_df = detect_anomalies(predictions_df)

        results = []
        for i, row in predictions_df.iterrows():
            tail = row[TAIL_NUMBER_COL] if TAIL_NUMBER_COL in row else f"Flight_{i}"
            results.append({
                "TAIL_NUMBER": tail,
                "CARRIER_DELAY": row["CARRIER_DELAY"],
                "WEATHER_DELAY": row["WEATHER_DELAY"],
                "SECURITY_DELAY": row["SECURITY_DELAY"],
                "Total_Predicted_Delay": row["Total_Predicted_Delay"],
                "Anomalies": anomalies_df[anomalies_df[TAIL_NUMBER_COL]==tail].to_dict(orient='records')[0],
                "Optimization_Suggestions": optimize_suggestions(row)
            })
        return {"status": "success", "predictions": results}

    except Exception as e:
        return {"status": "error", "message": str(e)}

# Existing flight info endpoint
@app.get("/get-flight-info/{tail_number}")
def get_flight_info(tail_number: str):
    if predictions_df.empty:
        return {"error": "No predictions available. Please upload a CSV first."}
    row = predictions_df[predictions_df[TAIL_NUMBER_COL]==tail_number]
    if row.empty:
        return {"error": "Tail number not found."}
    row = row.iloc[0]
    anomalies = anomalies_df[anomalies_df[TAIL_NUMBER_COL]==tail_number].iloc[0].to_dict()
    optimization = optimize_suggestions(row)
    return {
        "TAIL_NUMBER": row[TAIL_NUMBER_COL],
        "CARRIER_DELAY": row["CARRIER_DELAY"],
        "WEATHER_DELAY": row["WEATHER_DELAY"],
        "SECURITY_DELAY": row["SECURITY_DELAY"],
        "Total_Predicted_Delay": row["Total_Predicted_Delay"],
        "Anomalies": anomalies,
        "Optimization_Suggestions": optimization
    }
