import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from prophet import Prophet

# ==============================
# 1.1 LOAD HISTORICAL FLIGHT DATA (STRICT MODE)
# ==============================

file_path = "flight filtered_column.csv"
df = pd.read_csv(file_path)

#  FIX: Normalize column names ONCE
df.columns = df.columns.str.upper()

print("\nColumns detected:", df.columns.tolist())

# ------------------------------
# REQUIRED SCHEMA FOR TRAINING
# ------------------------------
required_columns_full_date = ["DATE"]
required_columns_split_date = ["YEAR", "MONTH", "DAY"]

required_other_columns = [
    "DAY_OF_WEEK",
    "AIRLINE",
    "FLIGHT_NUMBER",
    "TAIL_NUMBER",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "SCHEDULED_DEPARTURE",
    "DEPARTURE_TIME",
    "DEPARTURE_DELAY",
    "DISTANCE",
    "SCHEDULED_ARRIVAL",
    "ARRIVAL_TIME",
    "ARRIVAL_DELAY",
    "CANCELLED",
    "CANCELLATION_REASON",
    "AIR_SYSTEM_DELAY",
    "SECURITY_DELAY",
    "AIRLINE_DELAY",
    "LATE_AIRCRAFT_DELAY",
    "WEATHER_DELAY"
]

# ==============================
# 1.2 VERIFY REQUIRED COLUMNS
# ==============================
cols = df.columns.tolist()

has_full_date = all(col in cols for col in required_columns_full_date)
has_split_date = all(col in cols for col in required_columns_split_date)

if not has_full_date and not has_split_date:
    raise ValueError("Training CSV must contain DATE or YEAR+MONTH+DAY")

missing_other_cols = [c for c in required_other_columns if c not in cols]
if missing_other_cols:
    raise ValueError(f"Missing required training columns: {missing_other_cols}")

# ==============================
# 1.3 CREATE STANDARDIZED DATE COLUMN
# ==============================
if has_full_date:
    print("→ Using DATE column")
    df["DATE_STD"] = pd.to_datetime(df["DATE"], errors="coerce")

else:
    print("→ Combining YEAR + MONTH + DAY into Date")
    df["DATE_STD"] = pd.to_datetime(
        df[["YEAR", "MONTH", "DAY"]].rename(
            columns={"YEAR": "year", "MONTH": "month", "DAY": "day"}
        ), 
        errors="coerce"
    )

df = df.sort_values("DATE_STD").reset_index(drop=True)

# ==============================
# 1.4 CREATE ROUTE COLUMN
# ==============================
df["ROUTE"] = df["ORIGIN_AIRPORT"].astype(str) + "-" + df["DESTINATION_AIRPORT"].astype(str)

print("\nTraining dataset ready. First rows:")
print(df.head())

# Keep a copy of the fully preprocessed historical dataset before any further filtering
df_hist_all = df.copy()

# ==============================
# 1.5 SAFE ROUTE → AIRLINE SELECTION
# ==============================


all_routes = sorted(df["ROUTE"].unique())
print("\nTotal unique routes in dataset:", len(all_routes))
auto_select = os.environ.get("AUTO_SELECT_FIRST") == "1"

if auto_select:
    query = ""
else:
    query = input("Search route by substring (e.g., 'JFK' or 'LAX'), or press Enter to show sample: ").strip().upper()

if query:
    matches = [r for r in all_routes if query in r]
    if not matches:
        print(f"No routes matching '{query}' found. Showing sample instead.")
        matches = all_routes[:50]
    else:
        print(f"\nFound {len(matches)} matching routes (showing max 50):")
    print(matches[:50])
else:
    matches = all_routes[:50]
    print("\nShowing sample routes:")
    print(matches)

if auto_select:
    # choose the first match for automated runs
    ROUTE_TO_USE = matches[0] if matches else all_routes[0]
    print(f"Auto-selected route: {ROUTE_TO_USE}")
else:
    # interactive selection: allow substring or exact route
    while True:
        ROUTE_TO_USE = input("Enter exact route to forecast from the list above: ").strip().upper()
        if ROUTE_TO_USE in all_routes:
            break
        # allow user to enter an index when they typed a substring earlier
        possible = [r for r in all_routes if ROUTE_TO_USE in r]
        if len(possible) == 1:
            ROUTE_TO_USE = possible[0]
            print(f"Interpreting input as route '{ROUTE_TO_USE}'")
            break
        if possible:
            print("Multiple matches found. Showing up to 50 matches:")
            for i, r in enumerate(possible[:50]):
                print(f"{i}: {r}")
            sel = input("Enter index of route to choose, or press Enter to try again: ").strip()
            if sel.isdigit() and 0 <= int(sel) < len(possible[:50]):
                ROUTE_TO_USE = possible[int(sel)]
                break
            else:
                print("Please try entering a different substring or an exact route.")
                continue
        print(f"Route '{ROUTE_TO_USE}' does not exist in the dataset. Try again.")



valid_airlines = sorted(df[df["ROUTE"] == ROUTE_TO_USE]["AIRLINE"].unique())

print(f"\nAirlines flying {ROUTE_TO_USE}: {valid_airlines}")

if auto_select:
    AIRLINE_TO_USE = valid_airlines[0]
    print(f"Auto-selected airline: {AIRLINE_TO_USE}")
else:
    while True:
        AIRLINE_TO_USE = input(f"Choose airline from {valid_airlines}: ").strip().upper()

        if AIRLINE_TO_USE in valid_airlines:
            break

        print(f"\nInvalid airline '{AIRLINE_TO_USE}'. Please choose from: {valid_airlines}")

print(f"\n✔ Using airline '{AIRLINE_TO_USE}' and route '{ROUTE_TO_USE}'.\n")

# Apply filter early to avoid crash in forecast logic
df = df[(df["AIRLINE"] == AIRLINE_TO_USE) & (df["ROUTE"] == ROUTE_TO_USE)]

if df.empty:
    raise ValueError(
        f"\nNo historical data for airline '{AIRLINE_TO_USE}' on route '{ROUTE_TO_USE}'.\n"
        f"Please select a different route."
    )


# ==============================
# 1.6 ENCODE CATEGORICAL VARIABLES
# ==============================
df["AIRLINE_ENCODED"] = df["AIRLINE"].astype("category").cat.codes
df["ROUTE_ENCODED"]   = df["ROUTE"].astype("category").cat.codes

# ==============================
# 2 FEATURE ENGINEERING
# ==============================

# --- Standard uppercase feature names ---
df["DAY_OF_WEEK"] = df["DATE_STD"].dt.dayofweek
df["WEEK_OF_YEAR"] = df["DATE_STD"].dt.isocalendar().week.astype(int)
df["MONTH"] = df["DATE_STD"].dt.month
df["IS_WEEKEND"] = (df["DAY_OF_WEEK"] >= 5).astype(int)

# --- Lag features ---
for lag in [1,2,3,7,14]:
    df[f"delay_lag_{lag}"] = df.groupby(["AIRLINE","ROUTE"])["ARRIVAL_DELAY"].shift(lag)

# --- Differences ---
df["diff_1"] = df.groupby(["AIRLINE","ROUTE"])["ARRIVAL_DELAY"].diff(1)
df["diff_7"] = df.groupby(["AIRLINE","ROUTE"])["ARRIVAL_DELAY"].diff(7)

# --- Rolling windows ---
df["rolling_mean_7"] = df.groupby(["AIRLINE","ROUTE"])["ARRIVAL_DELAY"].shift(1).rolling(7).mean()
df["rolling_std_7"] = df.groupby(["AIRLINE","ROUTE"])["ARRIVAL_DELAY"].shift(1).rolling(7).std()
df["rolling_mean_30"] = df.groupby(["AIRLINE","ROUTE"])["ARRIVAL_DELAY"].shift(1).rolling(30).mean()

# Instead of dropping all rows with any NA (which removes almost all history
# because lag/rolling features introduce NaNs), selectively fill only the
# derived columns so we retain historical rows for training.

# Ensure target exists and fill with forward-fill then zero as a last resort
if "ARRIVAL_DELAY" in df.columns:
    df["ARRIVAL_DELAY"] = df["ARRIVAL_DELAY"].fillna(method="ffill").fillna(0)

# Fill lag columns with 0 where they are NaN (no history for the lag)
lag_cols = [f"delay_lag_{lag}" for lag in [1,2,3,7,14]]
for col in lag_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# Fill diffs with 0 if missing
for col in ["diff_1", "diff_7"]:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# For rolling statistics, fill with the column mean when possible, else 0
for col in ["rolling_mean_7", "rolling_std_7", "rolling_mean_30"]:
    if col in df.columns:
        if df[col].notna().any():
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(0)

# For other numeric columns used as features, fill NaNs with 0 to avoid training errors
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
for col in numeric_cols:
    df[col] = df[col].fillna(0)

df = df.reset_index(drop=True)

# ==============================
# 3️⃣ DEFINE FEATURES AND TARGET
# ==============================
feature_cols = [
    "WEATHER_DELAY", "AIR_SYSTEM_DELAY", "SECURITY_DELAY", "AIRLINE_DELAY",
    "LATE_AIRCRAFT_DELAY", "SCHEDULED_DEPARTURE",
    "SCHEDULED_ARRIVAL", "DISTANCE", "DEPARTURE_DELAY",
    "DAY_OF_WEEK", "WEEK_OF_YEAR", "MONTH", "IS_WEEKEND",
    "AIRLINE_ENCODED", "ROUTE_ENCODED",
    "diff_1", "diff_7",
]
feature_cols += [f"delay_lag_{lag}" for lag in [1,2,3,7,14]]
feature_cols += ["rolling_mean_7", "rolling_std_7", "rolling_mean_30"]

target = "ARRIVAL_DELAY"

X = df[feature_cols]
y = df[target]

# ==============================
# 4️⃣ TRAIN-TEST SPLIT
# ==============================
split_index = int(len(df)*0.8)
X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# ==============================
# 5️⃣ TRAIN XGBOOST MODEL (robust to empty training data)
# ==============================
if X_train.shape[0] == 0:
    print("\n⚠️ Training data is empty after feature engineering. Using fallback zero-model.")
    class ZeroModel:
        def predict(self, X):
            return np.zeros(len(X))
    model = ZeroModel()
else:
    model = XGBRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=12,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=1,
        reg_lambda=2,
        min_child_weight=5,
        objective="reg:squarederror"
    )
    model.fit(X_train, y_train)

# ==============================
# 6️⃣ SINGLE-FILE FUTURE ROWS (use the same uploaded CSV for future schedules)
# ==============================

# Single-file mode: do not load a second CSV. Use the original file specified
# by `file_path` (or the in-memory `df`) to find scheduled/future rows.


# --- Replace separate future-CSV logic: use the same uploaded CSV used for training ---
# Read the original training CSV path (`file_path`) to find scheduled rows (missing ARRIVAL_DELAY)
try:
    full_df = pd.read_csv(file_path)
    full_df.columns = full_df.columns.str.upper()
except Exception:
    # fallback to in-memory historical dataframe if reading fails
    full_df = df.copy()

# Ensure DATE_STD exists on full_df
if "DATE_STD" not in full_df.columns:
    if "DATE" in full_df.columns:
        full_df["DATE_STD"] = pd.to_datetime(full_df["DATE"], errors="coerce")
    elif all(c in full_df.columns for c in ["YEAR", "MONTH", "DAY"]):
        full_df["DATE_STD"] = pd.to_datetime(full_df[["YEAR", "MONTH", "DAY"]].rename(columns={"YEAR":"year","MONTH":"month","DAY":"day"}), errors="coerce")

# Normalize and build ROUTE on full_df
if "ORIGIN_AIRPORT" in full_df.columns and "DESTINATION_AIRPORT" in full_df.columns:
    full_df["ORIGIN_AIRPORT"] = full_df["ORIGIN_AIRPORT"].astype(str).str.upper()
    full_df["DESTINATION_AIRPORT"] = full_df["DESTINATION_AIRPORT"].astype(str).str.upper()
    full_df["ROUTE"] = full_df["ORIGIN_AIRPORT"] + "-" + full_df["DESTINATION_AIRPORT"]

if "AIRLINE" in full_df.columns:
    full_df["AIRLINE"] = full_df["AIRLINE"].astype(str).str.upper()

# Find scheduled rows in the same CSV matching airline+route (ARRIVAL_DELAY is NaN)
if "ARRIVAL_DELAY" in full_df.columns:
    matching_user_rows = full_df[(full_df["AIRLINE"] == AIRLINE_TO_USE) & (full_df["ROUTE"] == ROUTE_TO_USE) & (full_df["ARRIVAL_DELAY"].isna())]
else:
    matching_user_rows = full_df[(full_df.get("AIRLINE", "").str.upper() == AIRLINE_TO_USE) & (full_df.get("ROUTE", "") == ROUTE_TO_USE)].copy()

if matching_user_rows.empty:
    print(f"\nNo scheduled/future rows found in uploaded CSV for {AIRLINE_TO_USE} {ROUTE_TO_USE} — falling back to last historical row.")
    df_to_predict_on = df.tail(1).copy()
    using_user_input = False
else:
    print(f"\nUsing {len(matching_user_rows)} scheduled rows from the uploaded CSV for prediction.")
    df_to_predict_on = matching_user_rows.copy()
    using_user_input = True

# Ensure encoded columns exist for prediction rows
if "AIRLINE_ENCODED" not in df_to_predict_on.columns or "ROUTE_ENCODED" not in df_to_predict_on.columns:
    if not df.empty:
        hist_ref = df.iloc[-1]
        df_to_predict_on["AIRLINE_ENCODED"] = df_to_predict_on.get("AIRLINE_ENCODED", hist_ref.get("AIRLINE_ENCODED", 0))
        df_to_predict_on["ROUTE_ENCODED"] = df_to_predict_on.get("ROUTE_ENCODED", hist_ref.get("ROUTE_ENCODED", 0))
    else:
        df_to_predict_on["AIRLINE_ENCODED"] = df_to_predict_on.get("AIRLINE_ENCODED", 0)
        df_to_predict_on["ROUTE_ENCODED"] = df_to_predict_on.get("ROUTE_ENCODED", 0)

# ==============================
# 7️⃣ ASK USER FOR FORECAST DAYS
# ==============================
# Allow non-interactive runs by reading forecast days from env var `FORECAST_DAYS`
days_env = os.environ.get("FORECAST_DAYS") or os.environ.get("DAYS")
if days_env:
    try:
        days = int(days_env)
        print(f"Auto-using FORECAST_DAYS={days}")
    except Exception:
        days = int(input("How many future days to forecast? "))
else:
    days = int(input("How many future days to forecast? "))
# (future matching handled above using the single uploaded CSV)

# ==============================
# 8️⃣ FORECAST FUNCTIONS (FIXED)
# ==============================

def forecast_future_days_prophet(df, airline, route, n_days=365):
    # Select rows for this airline+route and aggregate to daily mean
    df_target = df[(df["AIRLINE"] == airline) & (df["ROUTE"] == route)][["DATE_STD","ARRIVAL_DELAY"]].copy()
    if df_target.empty:
        raise ValueError("No history for selected airline + route for Prophet.")

    df_target = df_target.rename(columns={"DATE_STD":"ds","ARRIVAL_DELAY":"y"})
    # Aggregate by day to produce a single time series point per day
    df_target = df_target.groupby("ds", as_index=False)["y"].mean().sort_values("ds")
    # Robustify Prophet input: winsorize historical y to remove extreme outliers and
    # use logistic growth with cap/floor derived from historical percentiles. This
    # prevents explosive extrapolation and produces more realistic forecasts.
    lo_pct = float(os.environ.get("PROPHET_INPUT_LO_PCT", "0.01"))
    hi_pct = float(os.environ.get("PROPHET_INPUT_HI_PCT", "0.99"))
    y_lo = df_target["y"].quantile(lo_pct)
    y_hi = df_target["y"].quantile(hi_pct)

    # winsorize
    df_target["y"] = df_target["y"].clip(lower=y_lo, upper=y_hi)

    # Use logistic growth with cap/floor set to historical percentiles
    df_target["cap"] = y_hi
    df_target["floor"] = y_lo

    cps = float(os.environ.get("PROPHET_CHANGPOINT_PRIOR_SCALE", "0.01"))
    model = Prophet(growth='logistic', daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True,
                    changepoint_prior_scale=cps)
    try:
        model.fit(df_target)
    except Exception:
        # Fallback: try without logistic growth if that fails
        model = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True,
                        changepoint_prior_scale=cps)
        model.fit(df_target.rename(columns={"cap":"cap"}))

    future = model.make_future_dataframe(periods=n_days)
    # ensure cap/floor exist on future for logistic growth
    if "cap" in df_target.columns:
        future["cap"] = y_hi
    if "floor" in df_target.columns:
        future["floor"] = y_lo

    forecast = model.predict(future)

    # Preserve raw Prophet output
    raw = forecast[["ds","yhat"]].tail(n_days).rename(columns={"ds":"DATE_STD","yhat":"yhat_raw"}).reset_index(drop=True)

    # Clip by robust percentiles (less likely to collapse all values)
    lo_q = float(os.environ.get("PROPHET_CLIP_LO", "0.01"))
    hi_q = float(os.environ.get("PROPHET_CLIP_HI", "0.99"))
    lower = df_target["y"].quantile(lo_q)
    upper = df_target["y"].quantile(hi_q)

    raw["PREDICTED_DELAY"] = raw["yhat_raw"].clip(lower=lower, upper=upper)

    # Diagnostics
    print(f"Prophet raw yhat min/max: {raw['yhat_raw'].min():.3f}/{raw['yhat_raw'].max():.3f}")
    print(f"Input winsorize bounds: {y_lo:.3f}/{y_hi:.3f}; Clipping bounds: {lower:.3f}/{upper:.3f}")
    print(f"Rows clipped to lower: {(raw['yhat_raw'] < lower).sum()}, to upper: {(raw['yhat_raw'] > upper).sum()}")

    return raw[["DATE_STD","PREDICTED_DELAY","yhat_raw"]]


def forecast_future_days(df, model, feature_cols, airline, route, n_days=365):
    df_target = df[(df["AIRLINE"] == airline) & (df["ROUTE"] == route)].copy()
    df_target = df_target.sort_values("DATE_STD").reset_index(drop=True)

    if df_target.empty:
        raise ValueError("No history for selected airline + route.")

    # --- lag preparation (FIXED) ---
    max_lag = 14

    # If ARRIVAL_DELAY doesn't exist OR is all NaN, fill with 0
    if "ARRIVAL_DELAY" not in df_target.columns or df_target["ARRIVAL_DELAY"].isna().all():
        df_target["ARRIVAL_DELAY"] = 0

    # Guarantee at least `max_lag` rows for stable lags
    if len(df_target) < max_lag + 1:
        # pad synthetic history
        pad_rows = (max_lag + 1) - len(df_target)
        pad_df = df_target.head(1).copy()
        pad_df = pd.concat([pad_df] * pad_rows, ignore_index=True)
        df_target = pd.concat([pad_df, df_target], ignore_index=True)

    # Create lag columns safely
    for lag in range(1, max_lag + 1):
        df_target[f"delay_lag_{lag}"] = df_target["ARRIVAL_DELAY"].shift(lag).fillna(0)


    last_row = df_target.iloc[-1].to_dict()
    lag_values = {lag: last_row.get(f"delay_lag_{lag}", 0) for lag in range(1, max_lag+1)}

    rolling7 = list(df_target["ARRIVAL_DELAY"].iloc[-7:])
    rolling30 = list(df_target["ARRIVAL_DELAY"].iloc[-30:])

    future_rows = []
    reference_row = last_row.copy()

    for _ in range(n_days):

        next_date = reference_row["DATE_STD"] + pd.Timedelta(days=1)
        new_row = {}

        # --- time features (uppercase) ---
        new_row["DAY_OF_WEEK"] = next_date.dayofweek
        new_row["WEEK_OF_YEAR"] = int(next_date.isocalendar().week)
        new_row["MONTH"] = next_date.month
        new_row["IS_WEEKEND"] = int(next_date.dayofweek >= 5)

        # --- static features (uppercase names) ---
        for col in [
            "WEATHER_DELAY", "AIR_SYSTEM_DELAY", "SECURITY_DELAY",
            "AIRLINE_DELAY", "LATE_AIRCRAFT_DELAY",
            "SCHEDULED_DEPARTURE", "SCHEDULED_ARRIVAL",
            "DISTANCE", "DEPARTURE_DELAY"
        ]:
            new_row[col] = reference_row.get(col, 0)

        # --- lag features ---
        new_row["delay_lag_1"] = lag_values[1]
        for lag in range(2, max_lag+1):
            new_row[f"delay_lag_{lag}"] = lag_values[lag]

        # --- diff ---
        new_row["diff_1"] = new_row["delay_lag_1"] - lag_values.get(2, new_row["delay_lag_1"])
        new_row["diff_7"] = new_row["delay_lag_1"] - lag_values.get(7, new_row["delay_lag_1"])

        # --- rolling ---
        rolling7.append(new_row["delay_lag_1"])
        rolling30.append(new_row["delay_lag_1"])
        rolling7 = rolling7[-7:]
        rolling30 = rolling30[-30:]

        new_row["rolling_mean_7"] = sum(rolling7) / len(rolling7)
        new_row["rolling_std_7"] = float(pd.Series(rolling7).std())
        new_row["rolling_mean_30"] = sum(rolling30) / len(rolling30)

        # --- encoded ---
        new_row["AIRLINE_ENCODED"] = reference_row["AIRLINE_ENCODED"]
        new_row["ROUTE_ENCODED"] = reference_row["ROUTE_ENCODED"]

        # --- model prediction ---
        X_pred = pd.DataFrame([[new_row.get(col, 0) for col in feature_cols]], columns=feature_cols)
        pred = float(model.predict(X_pred)[0])
        # include feature context in returned row for explanation
        row_out = new_row.copy()
        row_out.update({"DATE_STD": next_date, "PREDICTED_DELAY": pred})
        future_rows.append(row_out)

        # --- update lags ---
        new_lag_values = {1: pred}
        for lag in range(2, max_lag+1):
            new_lag_values[lag] = lag_values[lag - 1]
        lag_values = new_lag_values.copy()

        # --- update reference for next iteration ---
        reference_row["DATE_STD"] = next_date
        reference_row["ARRIVAL_DELAY"] = pred
        for k in new_row:
            reference_row[k] = new_row[k]

    # ensure consistent column order: date, predicted, then features
    df_future = pd.DataFrame(future_rows)
    cols = [c for c in ["DATE_STD", "PREDICTED_DELAY"] + list(df_future.columns) if c in df_future.columns]
    # remove duplicates while preserving order
    seen = set(); cols_unique = []
    for c in cols:
        if c not in seen:
            cols_unique.append(c); seen.add(c)
    return df_future[cols_unique]
# ==============================
# AUTO-SET RISK THRESHOLD
# ==============================
# Example: threshold = mean + 1 std of historical delays
threshold_for_forecast = df["ARRIVAL_DELAY"].mean() + df["ARRIVAL_DELAY"].std()


# ==============================
# 9️⃣ DETECT HIGH-RISK FLIGHTS 
# ==============================
def detect_future_risks(forecast_df, threshold, hist_df=None):
    df = forecast_df.copy()

    # Use historical dataframe if not provided (falls back to df_hist_all if we have it)
    if hist_df is None:
        hist_df = globals().get("df_hist_all", None)

    if hist_df is not None and "ARRIVAL_DELAY" in hist_df.columns:
        hist_mean = hist_df["ARRIVAL_DELAY"].mean()
        hist_std = hist_df["ARRIVAL_DELAY"].std()
    else:
        hist_mean = 0.0
        hist_std = 1.0

    # Configurable multiplier for negative (early arrival) anomaly threshold
    try:
        neg_k = float(os.environ.get("RISK_NEG_K", "3"))
    except Exception:
        neg_k = 3.0

    neg_threshold = hist_mean - (neg_k * hist_std)

    # New rule: flag any positive predicted delay (>0). Flag large early arrivals only when
    # predicted < neg_threshold (i.e., much earlier than historical mean by neg_k stddevs).
    def flag_pred(p):
        try:
            p = float(p)
        except Exception:
            return False
        if p > 0:
            return True
        if p < neg_threshold:
            return True
        return False

    df["RISK"] = df["PREDICTED_DELAY"].apply(flag_pred)

    static_factors = [
        "WEATHER_DELAY", "AIR_SYSTEM_DELAY", "SECURITY_DELAY",
        "AIRLINE_DELAY", "LATE_AIRCRAFT_DELAY", "DEPARTURE_DELAY"
    ]

    hist_medians = {}
    if hist_df is not None:
        for f in static_factors:
            if f in hist_df.columns:
                hist_medians[f] = hist_df[f].median()

    def explain_row(row):
        pred = row.get("PREDICTED_DELAY", 0.0)
        reasons = []
        for f in static_factors:
            val = row.get(f, None)
            if val is None:
                continue
            if val > 0:
                reasons.append(f)
            else:
                med = hist_medians.get(f)
                if med is not None and val > med:
                    reasons.append(f)

        reasons = list(dict.fromkeys(reasons))

        # Use different wording for early-arrival risks
        if row.get("PREDICTED_DELAY", 0) < 0 and row.get("RISK", False):
            base_msg = f"Predicted early arrival ~{abs(row.get('PREDICTED_DELAY',0)):.0f} minutes."
        else:
            base_msg = f"Predicted delay ~{row.get('PREDICTED_DELAY',0):.0f} minutes."

        if reasons:
            return base_msg + " Likely contributing factors: " + ", ".join(reasons)
        else:
            return base_msg + " Contributing factors: unknown or not in input CSV."

    df["MESSAGE"] = df.apply(lambda r: explain_row(r), axis=1)

    return df[[c for c in ["DATE_STD", "PREDICTED_DELAY", "RISK", "MESSAGE"] if c in df.columns]]

# ==============================
# 10️⃣ GENERATE FORECAST
# ==============================
min_hist_days = int(os.environ.get("PROPHET_MIN_HISTORY_DAYS", "30"))

if days < 15:
    forecast_result = forecast_future_days(
        df_to_predict_on,
        model,
        feature_cols,
        AIRLINE_TO_USE,
        ROUTE_TO_USE,
        n_days=days
    )

else:
    # Check whether we have enough historical days for Prophet; if not, fall back to the
    # iterative XGBoost-based generator which tends to be more stable on short histories.
    hist_for_prophet = df[(df["AIRLINE"] == AIRLINE_TO_USE) & (df["ROUTE"] == ROUTE_TO_USE)][["DATE_STD","ARRIVAL_DELAY"]].dropna()
    unique_days = hist_for_prophet["DATE_STD"].nunique()
    if unique_days < min_hist_days:
        print(f"Insufficient history for Prophet ({unique_days} days < {min_hist_days}). Falling back to iterative forecast.")
        forecast_result = forecast_future_days(
            df_to_predict_on,
            model,
            feature_cols,
            AIRLINE_TO_USE,
            ROUTE_TO_USE,
            n_days=days
        )
    else:
        forecast_result = forecast_future_days_prophet(
            df,
            AIRLINE_TO_USE,
            ROUTE_TO_USE,
            n_days=days
        )

        # Safety check: if Prophet produced extreme raw values or collapsed to a single clipped
        # value, fallback to the iterative XGBoost generator which is more stable on odd trends.
        try:
            hist_std_check = df["ARRIVAL_DELAY"].std()
            if np.isnan(hist_std_check) or hist_std_check == 0:
                hist_std_check = 1.0

            is_extreme = False
            if "yhat_raw" in forecast_result.columns:
                max_abs = forecast_result["yhat_raw"].abs().max()
                if max_abs > (10 * hist_std_check):
                    is_extreme = True

            # Collapsed predictions (all equal) often mean clipping collapsed everything
            collapsed = forecast_result["PREDICTED_DELAY"].nunique() <= 1

            if is_extreme or collapsed:
                print("Prophet produced extreme or collapsed forecasts — falling back to iterative XGBoost forecast.")
                forecast_result = forecast_future_days(
                    df_to_predict_on,
                    model,
                    feature_cols,
                    AIRLINE_TO_USE,
                    ROUTE_TO_USE,
                    n_days=days
                )
        except Exception:
            # If anything goes wrong in the safeguard, prefer iterative generator to avoid catastrophes
            print("Error evaluating Prophet output; falling back to iterative forecast.")
            forecast_result = forecast_future_days(
                df_to_predict_on,
                model,
                feature_cols,
                AIRLINE_TO_USE,
                ROUTE_TO_USE,
                n_days=days
            )

print(f"\n===== {days}-DAY FLIGHT DELAY FORECAST =====")
print(forecast_result)

future_risks = detect_future_risks(forecast_result, threshold=threshold_for_forecast)

print("\n===== HIGH-RISK FLIGHTS =====")
print(future_risks[future_risks["RISK"]])

# ----------------------------
# Save outputs for downstream optimization or review
# ----------------------------
out_dir = "forecast_outputs"
os.makedirs(out_dir, exist_ok=True)

# create safe filename parts
safe_route = ROUTE_TO_USE.replace("/", "_").replace(" ", "")
safe_airline = AIRLINE_TO_USE.replace("/", "_").replace(" ", "")
ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

forecast_path = f"{out_dir}/forecast_{safe_airline}_{safe_route}_{ts}.csv"
risks_path = f"{out_dir}/risks_{safe_airline}_{safe_route}_{ts}.csv"

try:
    forecast_result.to_csv(forecast_path, index=False)
    future_risks.to_csv(risks_path, index=False)
    print(f"\nSaved forecast CSV: {forecast_path}")
    print(f"Saved risks CSV: {risks_path}")
except Exception as e:
    print(f"Failed to save CSVs: {e}")
