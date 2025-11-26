import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from tkinter import Tk, filedialog # For simulating file upload

# =============================================================
# CONFIGURATION & CONTEXT (MUST MATCH TRAINING DATA)
# =============================================================

# --- Input Files ---
MODEL_FILENAME = 'xgb_flight_delay_model_trained.joblib'
# Placeholder for testing, replace with user upload logic in a real UI
SCHEDULE_CSV_PATH = 'synthetic_schedule_1500.csv' 

# --- Context (Inferred from training data—used for filtering and thresholds) ---
# NOTE: These values should ideally be calculated from your full y_train distribution.
HIST_MEAN_DELAY = 5.0 # Placeholder for mean historical delay (minutes)
HIST_STD_DELAY = 20.0 # Placeholder for std dev historical delay (minutes)
RISK_NEG_K = 3.0 # Factor for defining a "large" early arrival (e.g., 3 standard deviations early)

# --- Define the 52+ feature names used by the model ---
FEATURE_COLS = [
    'YEAR', 'QUARTER', 'DAY_OF_MONTH', 'MARKETING_AIRLINE', 'MARKETING_FLIGHT_NUM', 
    'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_CODE', 'ORIGIN_STATE', 'DEST_AIRPORT_ID', 
    'DEST_AIRPORT_CODE', 'DEST_STATE', 'DEPARTURE_BLOCK', 'SCHEDULED_ARRIVAL_TIME', 
    'ARRIVAL_BLOCK', 'IS_CANCELLED', 'IS_DIVERTED', 'SCHEDULED_FLIGHT_DURATION', 
    'DISTANCE_GROUP_ID', 'DEP_HOUR_SIN', 'DEP_HOUR_COS', 'DEP_MIN_SIN', 'DEP_MIN_COS', 
    'MONTH_SIN', 'MONTH_COS', 'DAY_OF_WEEK_SIN', 'DAY_OF_WEEK_COS', 'DISTANCE_LOG', 
    'AIRLINE_9E', 'AIRLINE_AA', 'AIRLINE_AS', 'AIRLINE_AX', 'AIRLINE_B6', 
    'AIRLINE_C5', 'AIRLINE_CP', 'AIRLINE_DL', 'AIRLINE_EM', 'AIRLINE_EV', 
    'AIRLINE_F9', 'AIRLINE_G4', 'AIRLINE_G7', 'AIRLINE_HA', 'AIRLINE_MQ', 
    'AIRLINE_NK', 'AIRLINE_OH', 'AIRLINE_OO', 'AIRLINE_PT', 'AIRLINE_QX', 
    'AIRLINE_UA', 'AIRLINE_WN', 'AIRLINE_YV', 'AIRLINE_YX', 'AIRLINE_ZW'
]


# =============================================================
# CORE FUNCTIONS
# =============================================================

def load_model(filename: str):
    """Loads the trained XGBoost model."""
    try:
        model = joblib.load(filename)
        return model
    except FileNotFoundError:
        raise SystemExit(f"Error: Model file '{filename}' not found. Ensure it is in the script directory.")
    except Exception as e:
        raise SystemExit(f"Error loading model: {e}")

def preprocess_and_engineer(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstructs the 52+ features required by the trained model and adds 
    necessary display columns (ROUTE, AIRLINE, DATE_OF_FLIGHT).
    ⚠️ CRITICAL: This is a placeholder. You must replace this 
    section with the EXACT 52-feature construction logic from your training script.
    """
    df_proc = df_raw.copy()
    df_proc.columns = df_proc.columns.str.upper()

    # --- Feature Reconstruction Placeholder ---
    
    # 1. Ensure required feature columns are present (filling missing with 0 for safety)
    for col in FEATURE_COLS:
        if col not in df_proc.columns:
            df_proc[col] = 0.0

    # 2. Reconstruct categorical/route columns for filtering ant display
    
    # Check for existence and apply aggressive cleaning/upper-casing
    airline_col = df_proc.get('MARKETING_AIRLINE', pd.Series(''))
    orig_col = df_proc.get('ORIGIN_AIRPORT_CODE', pd.Series(''))
    dest_col = df_proc.get('DEST_AIRPORT_CODE', pd.Series(''))

    df_proc['AIRLINE'] = airline_col.astype(str).str.strip().str.upper()
    df_proc['ORIGIN_AIRPORT_CODE'] = orig_col.astype(str).str.strip().str.upper()
    df_proc['DEST_AIRPORT_CODE'] = dest_col.astype(str).str.strip().str.upper()

    # Reconstruct ROUTE
    df_proc['ROUTE'] = df_proc['ORIGIN_AIRPORT_CODE'] + "-" + df_proc['DEST_AIRPORT_CODE']
    
    # --- CRITICAL FIX: Filter out problematic codes immediately ---
    
    # Identify placeholder values that result from NaNs or mixed types being converted to string
    PLACEHOLDERS = {'0.0', 'UNK', 'UNKNOWN-ROUTE', 'NAN', ''}
    
    # Check 1: Route components must not be placeholders
    valid_mask_placeholders = (df_proc['ORIGIN_AIRPORT_CODE'].isin(PLACEHOLDERS) == False) & \
                              (df_proc['DEST_AIRPORT_CODE'].isin(PLACEHOLDERS) == False) & \
                              (df_proc['AIRLINE'].isin(PLACEHOLDERS) == False)
    
    # Check 2: Airport codes must be 3 characters long (filter out the numeric '0.0', '1.0', etc.)
    valid_mask_length = (df_proc['ORIGIN_AIRPORT_CODE'].str.len() == 3) & \
                        (df_proc['DEST_AIRPORT_CODE'].str.len() == 3)

    # Check 3: Airport codes must contain only letters (to eliminate accidental numeric/corrupted entries)
    valid_mask_alpha = (df_proc['ORIGIN_AIRPORT_CODE'].str.isalpha()) & \
                       (df_proc['DEST_AIRPORT_CODE'].str.isalpha())

    df_proc = df_proc[valid_mask_placeholders & valid_mask_length & valid_mask_alpha].reset_index(drop=True)
    
    if df_proc.empty:
        raise ValueError("All rows filtered out. Check CSV columns for valid 3-letter IATA Airport/Airline codes.")

    # 3. Add a date column placeholder for display/saving (using available date parts)
    # Ensure YEAR/MONTH/DAY_OF_MONTH are Series of correct length (defaults when missing)
    def _as_series_or_default(df, colname, default):
        val = df.get(colname, None)
        if val is None:
            return pd.Series([default] * len(df), index=df.index)
        # If a scalar slipped through (not a Series), make a Series of that scalar
        if not isinstance(val, (pd.Series, pd.Index)):
            return pd.Series([val] * len(df), index=df.index)
        return val

    year_s = _as_series_or_default(df_proc, 'YEAR', 2024).astype(str)
    month_s = _as_series_or_default(df_proc, 'MONTH', 1).astype(str).str.zfill(2)
    day_s = _as_series_or_default(df_proc, 'DAY_OF_MONTH', 1).astype(str).str.zfill(2)

    df_proc['DATE_OF_FLIGHT'] = pd.to_datetime(year_s + '-' + month_s + '-' + day_s, errors='coerce')
    df_proc['DATE_OF_FLIGHT'] = df_proc['DATE_OF_FLIGHT'].fillna(pd.Timestamp.now().normalize())

    # 4. Finalize data types and order for prediction
    df_pred_ready = df_proc[FEATURE_COLS].copy()
    
    for col in FEATURE_COLS:
        if df_pred_ready[col].dtype == 'object':
            # Converting string features to numeric 0 if they failed to be encoded
            df_pred_ready[col] = pd.to_numeric(df_pred_ready[col], errors='coerce').fillna(0).astype(np.int32)
        elif df_pred_ready[col].dtype != 'float32':
             df_pred_ready[col] = df_pred_ready[col].astype('float32')
            
    # Add back columns needed for display/filtering/saving
    df_pred_ready['AIRLINE'] = df_proc['AIRLINE']
    df_pred_ready['ROUTE'] = df_proc['ROUTE']
    df_pred_ready['SCHEDULED_ARRIVAL_TIME'] = df_proc.get('SCHEDULED_ARRIVAL_TIME', 0)
    df_pred_ready['DATE_OF_FLIGHT'] = df_proc['DATE_OF_FLIGHT']
    df_pred_ready['ORIGIN_AIRPORT_CODE'] = df_proc['ORIGIN_AIRPORT_CODE']
    df_pred_ready['DEST_AIRPORT_CODE'] = df_proc['DEST_AIRPORT_CODE']
    
    return df_pred_ready


def generate_future_rows(df: pd.DataFrame, n_extra: int) -> pd.DataFrame:
    """Append `n_extra` future rows by copying the last row and incrementing the date fields.
    This keeps route/airline/origin/dest and scheduled times the same and updates YEAR/MONTH/DAY_OF_MONTH/DATE_OF_FLIGHT.
    """
    if n_extra <= 0:
        return df

    # Ensure DATE_OF_FLIGHT is datetime
    df = df.copy()
    df['DATE_OF_FLIGHT'] = pd.to_datetime(df['DATE_OF_FLIGHT'])

    last_row = df.iloc[-1]
    future_rows = []
    for i in range(1, n_extra + 1):
        new_row = last_row.copy()
        new_date = last_row['DATE_OF_FLIGHT'] + pd.Timedelta(days=i)
        new_row['DATE_OF_FLIGHT'] = new_date
        # Update simple date parts that may be used by the model or display
        new_row['YEAR'] = float(new_date.year)
        new_row['MONTH'] = float(new_date.month)
        new_row['DAY_OF_MONTH'] = float(new_date.day)
        # Ensure all model FEATURE_COLS are numeric and have consistent dtypes
        for f in FEATURE_COLS:
            if f in new_row.index:
                # If the value is non-numeric (e.g., 'SFO' left in ORIGIN_AIRPORT_CODE),
                # coerce to numeric and fill with 0. This keeps the matrix numeric for XGBoost.
                try:
                    # Try converting to float first
                    new_row[f] = float(new_row[f])
                except Exception:
                    # Coerce strings to numeric where possible, else 0.0
                    try:
                        coerced = pd.to_numeric(pd.Series([new_row[f]]), errors='coerce').iloc[0]
                        if pd.isna(coerced):
                            new_row[f] = 0.0
                        else:
                            new_row[f] = float(coerced)
                    except Exception:
                        new_row[f] = 0.0
        future_rows.append(new_row)

    if future_rows:
        df_extra = pd.DataFrame(future_rows)
        df_out = pd.concat([df, df_extra], ignore_index=True)
        return df_out
    return df

def interactive_filter(df: pd.DataFrame):
    """
    Prompts user for route prefix using dynamic examples, lists matches, 
    and returns filtered data based on user input for route and airline.
    """
    
    all_routes = sorted(df['ROUTE'].unique())
    
    # Extract unique airport codes for dynamic examples
    unique_origins = df['ORIGIN_AIRPORT_CODE'].unique()
    unique_dests = df['DEST_AIRPORT_CODE'].unique()
    
    # Filter out UNK and combine codes
    all_codes = np.unique(np.concatenate([unique_origins, unique_dests]))
    clean_codes = [c for c in all_codes if c != 'UNK' and c != 'UNKNOWN-ROUTE' and len(c) == 3] # Assume standard 3-letter codes
    
    # --- Robust Example Selection ---
    
    # 1. Try to get 3 random clean codes
    if len(clean_codes) >= 3:
        np.random.seed(42) 
        example_codes = np.random.choice(clean_codes, 3, replace=False)
    # 2. If < 3 codes, use what's available
    elif len(clean_codes) > 0:
        example_codes = clean_codes
    # 3. Fallback to an empty list (DO NOT USE HARDCODED SFO/LAX)
    else:
        example_codes = [] 
    
    print("\n--- 3a. Filter Routes ---")
    
    # --- Route Selection ---
    selected_route = None
    while selected_route is None:
        # 1. Prompt for prefix with dynamic examples
        example_str = ", ".join([f"'{c}'" for c in example_codes])
        
        # NOTE: If no examples are available, example_str is empty, prompting like: "Example: "
        query = input(f"Enter route prefix/suffix (Example: {example_str}): ").strip().upper()
        
        # --- Check for matches ---
        matches = [r for r in all_routes if query in r]
        
        if not query:
            # If user presses Enter (no query), show sample
            matches = all_routes[:50]
            print(f"\nShowing sample routes ({len(matches)}):")
        elif not matches:
            # If search fails, inform user and SHOW THE AVAILABLE LIST
            print(f"No routes matching prefix/suffix '{query}' found.")
            
            # --- CORRECTED FLOW: Show available routes and ask for exact route ---
            print("\nPlease review the sample routes below to enter the exact match:")
            # Display sample of available routes from the entire set
            print(f"Available sample routes: {', '.join(all_routes[:5])}")
            # If search fails, we skip the print-matches step, but proceed to the exact route input
            
        else:
            # If matches found, print them
            print(f"\nMatching Routes ({len(matches)} total, showing max 50):")
            for i, r in enumerate(matches[:50]):
                print(f"  {i+1}: {r}")

        # 2. Prompt for exact route selection (This prompt always runs if matches were found or query was empty)
        route_input = input("Enter the EXACT route (e.g., 'SFO-LAX') to forecast: ").strip().upper()
        
        if route_input in all_routes:
            selected_route = route_input
        else:
            print("Invalid input. Please enter an exact route from the list.")
            
    # Filter by selected route
    df_route_filtered = df[df['ROUTE'] == selected_route]
    valid_airlines = sorted(df_route_filtered['AIRLINE'].unique())
    valid_airlines = [a for a in valid_airlines if a.strip()] # Remove empty/nan airlines

    # --- Airline Selection ---
    print("\n--- 3b. Select Airline ---")
    print(f"Airlines flying {selected_route}: {valid_airlines}")
    
    selected_airline = None
    while selected_airline is None:
        airline_input = input("Choose EXACT airline code (e.g., 'UA') to forecast: ").strip().upper()

        if airline_input in valid_airlines:
            selected_airline = airline_input
            break
        print(f"\nInvalid airline '{airline_input}'. Please choose from: {valid_airlines}")

    print(f"\n✔ Selected: Airline '{selected_airline}' on Route '{selected_route}'.")
    
    return df_route_filtered[df_route_filtered['AIRLINE'] == selected_airline].reset_index(drop=True), selected_airline, selected_route


def flag_anomalies(df_pred: pd.DataFrame) -> pd.DataFrame:
    """Flags anomalies so that:
    - Positive predicted delays are always flagged (`RISK=True`) and categorized
      into `RISK_LEVEL` = LOW / MEDIUM / HIGH using minute thresholds.
    - Negative predicted delays are left unchanged (not flagged).
    """

    # Risk thresholds (minutes)
    LOW_THRESH = 15.0   # 0 < delay <= 15 -> LOW
    MED_THRESH = 60.0   # 15 < delay <= 60 -> MEDIUM

    # Initialize columns
    df_pred['RISK'] = False
    df_pred['RISK_LEVEL'] = ""

    # Mask for positive (late) predictions
    pos_mask = df_pred['PREDICTED_DELAY'] > 0

    # Flag all positive delays
    df_pred.loc[pos_mask, 'RISK'] = True

    # Assign risk levels for positives
    df_pred.loc[pos_mask & (df_pred['PREDICTED_DELAY'] <= LOW_THRESH), 'RISK_LEVEL'] = 'LOW'
    df_pred.loc[pos_mask & (df_pred['PREDICTED_DELAY'] > LOW_THRESH) & \
                  (df_pred['PREDICTED_DELAY'] <= MED_THRESH), 'RISK_LEVEL'] = 'MEDIUM'
    df_pred.loc[pos_mask & (df_pred['PREDICTED_DELAY'] > MED_THRESH), 'RISK_LEVEL'] = 'HIGH'

    # Generate message: include risk level for positives, otherwise a neutral message
    def generate_message(row):
        pred = row['PREDICTED_DELAY']
        if row['RISK']:
            level = row['RISK_LEVEL'] if row['RISK_LEVEL'] else 'UNKNOWN'
            return f"Predicted DELAY of +{pred:.1f} min ({level} Risk)."
        else:
            return f"Predicted arrival: {pred:.1f} min (No Risk)."

    df_pred['MESSAGE'] = df_pred.apply(generate_message, axis=1)
    return df_pred


# =============================================================
# MAIN EXECUTION
# =============================================================

if __name__ == "__main__":
    
    # --- 1. Load Model ---
    model = load_model(MODEL_FILENAME)
    
    print("\nModel loaded successfully. Ready for prediction.")
    
    # --- 2. Load Scheduled CSV Data (Simulated Upload) ---
    print("\n--- 2. Loading Schedule Data ---")
    try:
        # Hide the Tkinter root window
        Tk().withdraw()

        # Let the user pick a file (defaults to Documents) — fallback to the hardcoded path
        file_path = filedialog.askopenfilename(initialdir=os.path.expanduser('~\\Documents'),
                                               title='Select schedule CSV',
                                               filetypes=[('CSV files', '*.csv'), ('All files', '*.*')])

        if not file_path:
            # No selection: fall back to the default used for testing
            file_path = SCHEDULE_CSV_PATH
            print(f"No file selected. Falling back to default: {file_path}")

        # NOTE: Using dtype=str ensures the airport codes are read as clean strings, preventing the 0.0 error.
        # Enforcing NA values to be empty strings helps robustness.
        df_raw_schedule = pd.read_csv(file_path, dtype=str, keep_default_na=False, na_values=[''])
        print(f"Loaded {len(df_raw_schedule)} total raw schedule rows from {file_path}.")

    except Exception as e:
        print(f"Error loading schedule data: {e}")
        raise SystemExit(1)

    # --- Preprocess Data ---
    try:
        # This will now filter out rows where airport codes are not clean 3-letter strings
        df_processed = preprocess_and_engineer(df_raw_schedule)
    except ValueError as e:
        print(f"\nCRITICAL ERROR during preprocessing: {e}")
        print("Please check your CSV file for valid airport and airline codes.")
        raise SystemExit(1)
        
    # --- 3. Filter Data by User Input (Interactive) ---
    df_filtered, selected_airline, selected_route = interactive_filter(df_processed)

    if df_filtered.empty:
        print("\nNo rows matched the selected criteria after filtering.")
        raise SystemExit(1)
        
    # --- 4. Prompt for number of future DAYS to forecast ---
    while True:
        try:
            # The user wants predictions for future days (e.g., next 7 days).
            ndays = int(input("\nHow many future DAYS to forecast (e.g., 7): ").strip())
            if ndays <= 0:
                print("Please enter an integer greater than 0.")
                continue

            # Generate exactly `ndays` future rows starting after the last available date
            print(f"Generating {ndays} future day rows based on the last available schedule row.")
            df_filtered = generate_future_rows(df_filtered, ndays)
            # We'll predict only the generated future rows (the last `ndays` rows)
            days_to_forecast = ndays
            break
        except ValueError:
            print("Invalid input. Please enter an integer.")

    # --- 5. Predict Delay ---
    
    # 1. Prepare data: predict only the future rows we just generated (last `days_to_forecast` rows)
    X_new = df_filtered[FEATURE_COLS].copy()
    X_new = X_new.apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32')

    # Select only the last `days_to_forecast` rows as the target prediction set
    X_pred = X_new.tail(days_to_forecast).copy()

    # 2. Run prediction on those future rows
    predictions = model.predict(X_pred)

    # 3. Attach predictions back to the corresponding rows in df_filtered
    df_filtered.loc[df_filtered.index[-days_to_forecast:], 'PREDICTED_DELAY'] = predictions

    # 4. Flag anomalies across the predicted slice
    df_final = flag_anomalies(df_filtered)
    
    # --- 6. Display and Slice Results for Output ---
    
    # Now, slice the final output dataframe to the generated future rows (last `days_to_forecast` rows)
    df_output = df_final.tail(days_to_forecast)
    anomalies = df_output[df_output['RISK']]

    print("\n===== FORECAST RESULTS =====")
    print(f"Airline: {selected_airline}, Route: {selected_route}")
    
    # Display the essential columns: Date and Predicted Delay
    output_cols_display = ['DATE_OF_FLIGHT', 'SCHEDULED_ARRIVAL_TIME', 'PREDICTED_DELAY', 'MESSAGE', 'RISK']
    print(df_output[output_cols_display].head(15).to_string())

    print("\n===== ANOMALY FLIGHTS (RISK=True) =====")

    if anomalies.empty:
        print("No high-risk flights or extreme anomalies detected in the displayed forecast.")
    else:
        print(f"Detected {len(anomalies)} anomalies in the displayed forecast. Showing first 5:")
        print(anomalies[output_cols_display].head(5).to_string())
        
    # --- 7. Save Results to CSV Files ---
    out_dir = "forecast_results"
    os.makedirs(out_dir, exist_ok=True)
    
    # Filename components
    safe_airline = selected_airline.replace('/', '_')
    safe_route = selected_route.replace('/', '_')
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # 7a. Save All Predicted Delays (Date | Predicted Delay)
    summary_cols = ['DATE_OF_FLIGHT', 'PREDICTED_DELAY']
    summary_path = os.path.join(out_dir, f"{safe_airline}_{safe_route}_summary_{ts}.csv")
    
    # SAVE ONLY THE REQUESTED NUMBER OF FLIGHTS
    df_output[summary_cols].to_csv(summary_path, index=False)
    print(f"\n✔ Predicted delays saved to: {summary_path}")

    # 7b. Save Anomalies (Risk)
    risk_cols = ['DATE_OF_FLIGHT', 'SCHEDULED_ARRIVAL_TIME', 'PREDICTED_DELAY', 'MESSAGE', 'RISK']
    risk_path = os.path.join(out_dir, f"{safe_airline}_{safe_route}_risks_{ts}.csv")
    
    # SAVE ONLY THE ANOMALIES FROM THE REQUESTED FORECAST LENGTH
    anomalies[risk_cols].to_csv(risk_path, index=False)
    print(f"✔ Anomalies/Risk data saved to: {risk_path}")
