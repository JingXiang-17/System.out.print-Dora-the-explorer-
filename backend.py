# backend.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np

app = FastAPI(title="Flight Delay & Optimization API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# GLOBAL DATAFRAME STORAGE
# -----------------------------
delay_df = pd.DataFrame()
risk_df = pd.DataFrame()
merged_df = pd.DataFrame()

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
# 1. CSV UPLOAD ENDPOINT
# -----------------------------
@app.post("/upload-csv")
async def upload_csv(delay_file: UploadFile = File(...), risk_file: UploadFile = File(...)):
    global delay_df, risk_df, merged_df
    try:
        delay_df = pd.read_csv(delay_file.file)
        risk_df = pd.read_csv(risk_file.file)
        merged_df = delay_df.merge(risk_df, on="TAIL_NUMBER", how="left")
        return {"status": "success", "message": f"{len(merged_df)} rows loaded."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# -----------------------------
# 2. ANOMALY DETECTION
# -----------------------------
def detect_anomalies(row):
    anomalies = []

    if row["CARRIER_DELAY"] > 15:
        anomalies.append(("Carrier Delay", carrier_suggestions))
    if row["WEATHER_DELAY"] > 10:
        anomalies.append(("Weather Delay", weather_suggestions))
    if row["SECURITY_DELAY"] > 5:
        anomalies.append(("Security Delay", security_suggestions))

    if row.get("CARRIER_DELAY_RISK", "LOW") in ("MEDIUM", "HIGH"):
        anomalies.append(("Carrier Risk", carrier_suggestions))
    if row.get("WEATHER_DELAY_RISK", "LOW") in ("MEDIUM", "HIGH"):
        anomalies.append(("Weather Risk", weather_suggestions))
    if row.get("SECURITY_DELAY_RISK", "LOW") in ("MEDIUM", "HIGH"):
        anomalies.append(("Security Risk", security_suggestions))

    return anomalies

# -----------------------------
# 3. OPTIMIZATION MODULE
# -----------------------------
def optimize_flight(row):
    suggestions = []

    if row["Total_Predicted_Delay"] > 20:
        suggestions.append("Consider earlier or later flights to reduce total delay.")
    if any(r in ("HIGH",) for r in [row.get("CARRIER_DELAY_RISK", "LOW"),
                                    row.get("WEATHER_DELAY_RISK", "LOW"),
                                    row.get("SECURITY_DELAY_RISK", "LOW")]):
        suggestions.append("High risk detected—consider flight alternatives or buffer time.")
    if row["WEATHER_DELAY"] > 10:
        suggestions.append("Buffer extra time for weather-related delays.")
    if row["SECURITY_DELAY"] > 5:
        suggestions.append("Arrive earlier to handle potential security delays.")

    return suggestions

# -----------------------------
# 4. GET FLIGHT INFO ENDPOINT
# -----------------------------
@app.get("/get-flight-info/{tail_number}")
def get_flight_info(tail_number: str):
    if merged_df.empty:
        return {"error": "No CSV uploaded yet."}

    row = merged_df[merged_df["TAIL_NUMBER"] == tail_number]
    if row.empty:
        return {"error": "TAIL_NUMBER not found"}

    row = row.iloc[0]
    anomalies = detect_anomalies(row)
    optimization = optimize_flight(row)

    return {
        "TAIL_NUMBER": row["TAIL_NUMBER"],
        "CARRIER_DELAY": row["CARRIER_DELAY"],
        "WEATHER_DELAY": row["WEATHER_DELAY"],
        "SECURITY_DELAY": row["SECURITY_DELAY"],
        "Total_Predicted_Delay": row["Total_Predicted_Delay"],
        "CARRIER_DELAY_RISK": row.get("CARRIER_DELAY_RISK", "LOW"),
        "WEATHER_DELAY_RISK": row.get("WEATHER_DELAY_RISK", "LOW"),
        "SECURITY_DELAY_RISK": row.get("SECURITY_DELAY_RISK", "LOW"),
        "Anomalies": [
            {"type": anomaly_type, "suggestions": suggestions}
            for anomaly_type, suggestions in anomalies
        ],
        "Optimization_Suggestions": optimization
    }
