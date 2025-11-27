import pandas as pd
import numpy as np
import joblib
import gc
import os 
from tkinter import Tk, filedialog

# --- CONFIGURATION (Must match training configuration) ---
TARGET_COLUMNS = ['CARRIER_DELAY', 'SECURITY_DELAY', 'WEATHER_DELAY']
MODEL_FILENAME = 'flight_delay_predictor.joblib'
USER_INPUT_FILE = 'user_input_flight_data_with_flightNumber.csv' 
TAIL_NUMBER_COL = 'TAIL_NUMBER' # <-- NEW: Identifier column to preserve

# Columns that MUST be present and converted to category for XGBoost inference
CATEGORICAL_COLS_FOR_XGB = [
    'ORIGIN_AIRPORT_CODE', 'DEST_AIRPORT_CODE', 'MARKETING_AIRLINE', 
    'ORIGIN_STATE', 'DEST_STATE', 'DEPARTURE_BLOCK', 'ARRIVAL_BLOCK'
]

def load_feature_names(filename='feature_names.txt'):
    """Loads the exact list of feature names the model was trained on."""
    # Try several locations: provided path, same directory as this script, then walk upwards
    # This makes the script robust when run from a different working directory.
    # 1) Exact path as given
    if os.path.exists(filename):
        path = filename
    else:
        # 2) Same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(script_dir, filename)
        if os.path.exists(candidate):
            path = candidate
        else:
            # 3) Walk the script directory tree to find the file
            path = None
            for root, dirs, files in os.walk(script_dir):
                if filename in files:
                    path = os.path.join(root, filename)
                    break
    if path is None:
        raise FileNotFoundError(f"feature names file '{filename}' not found. Searched cwd and '{script_dir}'.")
    with open(path, 'r') as f:
        feature_names = [line.strip() for line in f]
    return feature_names


def transform_data_for_inference(df_new, trained_feature_names):
    """
    Applies the necessary feature engineering and cleaning to new data
    to make it compatible with the trained model.
    """
    print("Applying inference transformations...")

    # 1. RENAME COLUMNS (Only if column names in user input are different)
    # Assuming user input uses the *original* names for simplicity in this demo.
    column_mapping = {
        'MKT_UNIQUE_CARRIER': 'MARKETING_AIRLINE',
        'OP_UNIQUE_CARRIER': 'OPERATING_AIRLINE', 
        'ORIGIN_AIRPORT_ID': 'ORIGIN_AIRPORT_ID', 'ORIGIN': 'ORIGIN_AIRPORT_CODE',
        'ORIGIN_STATE_ABR': 'ORIGIN_STATE', 'DEST_AIRPORT_ID': 'DEST_AIRPORT_ID',
        'DEST': 'DEST_AIRPORT_CODE', 'DEST_STATE_ABR': 'DEST_STATE',
        'CRS_DEP_TIME': 'SCHEDULED_DEPARTURE_TIME', 'DEP_TIME_BLK': 'DEPARTURE_BLOCK',
        'CRS_ARR_TIME': 'SCHEDULED_ARRIVAL_TIME', 'ARR_TIME_BLK': 'ARRIVAL_BLOCK',
        'CRS_ELAPSED_TIME': 'SCHEDULED_FLIGHT_DURATION', 'DISTANCE': 'FLIGHT_DISTANCE',
        'DISTANCE_GROUP': 'DISTANCE_GROUP_ID'
        # Crucially, we assume the user input does NOT contain leakage columns like ARR_DELAY
    }
    df_new = df_new.rename(columns=column_mapping)
    
    # 2. Impute/Clean (Fill any unexpected NaNs with the training set median/mode)
    # For simplicity, we use the median of the current new data, but ideally, 
    # you would use the stored medians from the training data.
    numeric_cols = df_new.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_new[col] = df_new[col].fillna(df_new[col].median())
        
    # 3. FEATURE ENGINEERING: TIME
    if 'SCHEDULED_DEPARTURE_TIME' in df_new.columns and 'MONTH' in df_new.columns:
        df_new['DEP_HOUR'] = (df_new['SCHEDULED_DEPARTURE_TIME'] // 100).astype(np.int8)
        df_new['DEP_MINUTE'] = (df_new['SCHEDULED_DEPARTURE_TIME'] % 100).astype(np.int8)
        
        df_new['DEP_HOUR_SIN'] = np.sin(2 * np.pi * df_new['DEP_HOUR'] / 24).astype(np.float32)
        df_new['DEP_HOUR_COS'] = np.cos(2 * np.pi * df_new['DEP_HOUR'] / 24).astype(np.float32)
        df_new['DEP_MIN_SIN'] = np.sin(2 * np.pi * df_new['DEP_MINUTE'] / 60).astype(np.float32)
        df_new['DEP_MIN_COS'] = np.cos(2 * np.pi * df_new['DEP_MINUTE'] / 60).astype(np.float32)
        
        for col in ['MONTH', 'DAY_OF_WEEK']:
            if col in df_new.columns:
                df_new[f'{col}_SIN'] = np.sin(2 * np.pi * df_new[col] / (12 if col=='MONTH' else 7)).astype(np.float32)
                df_new[f'{col}_COS'] = np.cos(2 * np.pi * df_new[col] / (12 if col=='MONTH' else 7)).astype(np.float32)

    # 4. FEATURE ENGINEERING: DISTANCE
    if 'FLIGHT_DISTANCE' in df_new.columns:
        df_new['DISTANCE_LOG'] = np.log1p(df_new['FLIGHT_DISTANCE']).astype(np.float32)

    # 5. Categorical Encoding (CRITICAL)

    # 5a. Convert all necessary string/object columns to 'category'
    for col in CATEGORICAL_COLS_FOR_XGB:
        if col in df_new.columns:
            df_new[col] = df_new[col].astype('category')
        else:
            # Handle missing essential columns by adding them with 'category' dtype and NaN values
            df_new[col] = np.nan
            df_new[col] = df_new[col].astype('category')

    # 5b. One-Hot Encoding for OPERATING_AIRLINE
    # This is complex in inference; for robustness, we use a fixed list of carrier columns.
    if 'OPERATING_AIRLINE' in df_new.columns:
        df_new = pd.get_dummies(df_new, columns=['OPERATING_AIRLINE'], prefix='AIRLINE', dtype='int8')

    # 6. Final Feature Alignment
    
    # 6a. Drop columns not needed for the final prediction
    cols_to_drop_post_engineering = [
        'SCHEDULED_DEPARTURE_TIME', 'DEP_HOUR', 'DEP_MINUTE', 'MONTH', 'DAY_OF_WEEK', 
        'SCHEDULED_ARRIVAL_TIME', 'FLIGHT_DISTANCE', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID'
    ]
    df_new = df_new.drop(columns=[c for c in cols_to_drop_post_engineering if c in df_new.columns], errors='ignore')

    # 6b. Align columns to match the trained model's feature order and names
    final_df = pd.DataFrame(index=df_new.index)
    
    for feature in trained_feature_names:
        if feature in df_new.columns:
            final_df[feature] = df_new[feature]
        else:
            # CRITICAL: If a feature used during training (like a specific AIRLINE_XXX column) 
            # is missing from the new data, add it with a value of 0.
            final_df[feature] = 0

    print(f"Inference transformation complete. Final features shape: {final_df.shape}")
    
    # Ensure the order and count of columns exactly match the training data
    assert len(final_df.columns) == len(trained_feature_names)
    assert all(final_df.columns == trained_feature_names)
    
    return final_df

if __name__ == "__main__":
    
    print("--- Simulation Start: Predicting New Flight Data ---")
    
    # 1. Load the trained feature names (CRITICAL)
    try:
        trained_feature_names = load_feature_names()
    except FileNotFoundError:
        print("\nERROR: feature_names.txt not found. Please run the training preprocessing script first.")
        exit()

    # 2. Let the user pick the raw CSV file via a file dialog (replaces automatic loading)
    Tk().withdraw()
    file_path = filedialog.askopenfilename(title='Select user CSV file', filetypes=[('CSV files', '*.csv'), ('All files', '*.*')])
    if not file_path:
        print("\nERROR: No file selected. Please select a CSV file to continue.")
        exit()
    try:
        df_raw = pd.read_csv(file_path)
        print(f"Successfully loaded data from {file_path}.")
        # --- Summary counts based on uploaded file ---
        total_rows = len(df_raw)
        print(f"Total rows in uploaded file: {total_rows}")

        # Flights under supervision: count unique tail numbers when available
        if TAIL_NUMBER_COL in df_raw.columns:
            unique_tails = int(df_raw[TAIL_NUMBER_COL].nunique())
            print(f"Number of flights under supervision (unique {TAIL_NUMBER_COL}): {unique_tails}")
        else:
            print(f"Column '{TAIL_NUMBER_COL}' not found; cannot compute unique tail numbers.")

        # Number of unique destinations (try common column names)
        dest_candidates = ['DEST', 'DEST_AIRPORT_CODE', 'DEST_AIRPORT_ID']
        dest_col = next((c for c in dest_candidates if c in df_raw.columns), None)
        if dest_col:
            unique_dests = int(df_raw[dest_col].nunique())
            print(f"Number of unique destinations ({dest_col}): {unique_dests}")
        else:
            print("No destination column found to compute unique destinations.")

        # Number of unique airlines (try common column names)
        airline_candidates = ['MARKETING_AIRLINE', 'MKT_UNIQUE_CARRIER', 'OPERATING_AIRLINE', 'OP_UNIQUE_CARRIER']
        airline_col = next((c for c in airline_candidates if c in df_raw.columns), None)
        if airline_col:
            unique_airlines = int(df_raw[airline_col].nunique())
            print(f"Number of unique airlines ({airline_col}): {unique_airlines}")
        else:
            print("No airline column found to compute unique airlines.")
    except Exception as e:
        print(f"\nERROR reading CSV: {e}")
        exit()

    # 3. SEPARATE ID COLUMN (NEW LOGIC)
    flight_ids = None
    if TAIL_NUMBER_COL in df_raw.columns:
        print(f"Separating and dropping ID column: {TAIL_NUMBER_COL}")
        # Save the ID column to merge back later
        flight_ids = df_raw[TAIL_NUMBER_COL].reset_index(drop=True)
        # Drop the ID from the DataFrame sent to the transformer
        df_raw = df_raw.drop(columns=[TAIL_NUMBER_COL])
    
    # 4. Transform the raw data (df_raw no longer contains the TAIL_NUMBER)
    X_inference = transform_data_for_inference(df_raw, trained_feature_names)
    
    # 5. Load the model and make predictions
    try:
        # If the model filename isn't an absolute path or not found in cwd, try to locate it
        model_path = MODEL_FILENAME
        if not os.path.exists(model_path):
            # search from script dir
            script_dir = os.path.dirname(os.path.abspath(__file__))
            found = None
            for root, dirs, files in os.walk(script_dir):
                if MODEL_FILENAME in files:
                    found = os.path.join(root, MODEL_FILENAME)
                    break
            if found:
                model_path = found
        model = joblib.load(model_path)
        
        print("\nMaking predictions...")
        y_pred = model.predict(X_inference)
        
        predictions_df = pd.DataFrame(y_pred, columns=TARGET_COLUMNS)
        predictions_df['Total_Predicted_Delay'] = predictions_df.sum(axis=1)
        
        # 6. MERGE ID BACK INTO RESULTS (NEW LOGIC)
        if flight_ids is not None:
            predictions_df.insert(0, TAIL_NUMBER_COL, flight_ids)
        
        print("\n--- Final Predictions (Minutes) ---")
        print(predictions_df)

        # Save predictions to fixed filename as requested
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(script_dir, 'forecast_results')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'predicted flight delay.csv')
        predictions_df.to_csv(out_path, index=False)
        print(f"\nSaved predictions CSV to: {out_path}")
        # --- Compute per-factor risk labels and save anomalies CSV ---
        def risk_label(value):
            av = abs(value)
            if av < 15:
                return 'LOW'
            if av <= 60:
                return 'MEDIUM'
            return 'HIGH'

        anomalies_df = pd.DataFrame()
        # Preserve identifier if present
        if TAIL_NUMBER_COL in predictions_df.columns:
            anomalies_df[TAIL_NUMBER_COL] = predictions_df[TAIL_NUMBER_COL]

        for col in TARGET_COLUMNS:
            risk_col = f"{col}_RISK"
            anomalies_df[risk_col] = predictions_df[col].apply(risk_label)

        anomalies_path = os.path.join(out_dir, 'flag anomalies.csv')
        anomalies_df.to_csv(anomalies_path, index=False)
        print(f"Saved anomalies CSV to: {anomalies_path}")

        # Print anomalies to terminal (same style as predictions)
        print("\n--- Flagged Anomalies ---")
        print(anomalies_df)
        
    except FileNotFoundError:
        print(f"\nERROR: Model file '{MODEL_FILENAME}' not found. Please train and save the model first.")
    
    gc.collect()
