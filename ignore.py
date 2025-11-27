import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import joblib # Added for model saving

# Ensure pyarrow is available to read parquet files
try:
    import pyarrow.parquet
except ImportError:
    print("FATAL: The 'pyarrow' library is required to read Parquet files. Please install it using 'pip install pyarrow'.")
    raise SystemExit(1)
    
# =============================================================
# 1. LOAD HISTORICAL FLIGHT DATA FROM PARQUET FILES (TRAINING)
# =============================================================

# Optional SHAP explainability
explainer = None

# --- Define Parquet file paths (use os.path.join to avoid escape issues) ---
X_train_path = os.path.join("dataset (1)", "X_train.parquet")
y_train_path = os.path.join("dataset (1)", "y_train.parquet")
X_test_path = os.path.join("dataset (1)", "X_test.parquet")

# --- Load Feature Matrices and Target ---
try:
    X_train = pd.read_parquet(X_train_path)
    y_train = pd.read_parquet(y_train_path).squeeze()
    X_test = pd.read_parquet(X_test_path)
    
except FileNotFoundError as e:
    raise FileNotFoundError(f"Missing required Parquet file: {e.filename}. Check paths and file names.")
except Exception as e:
    print(f"An error occurred while loading Parquet files: {e}")
    raise SystemExit(1)

print(f"\nTraining data loaded from Parquet:")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")

# Ensure columns are uppercase for consistency 
X_train.columns = X_train.columns.str.upper()
X_test.columns = X_test.columns.str.upper()

# Define feature_cols based on the loaded training data schema
feature_cols = X_train.columns.tolist()

# ---------- Ensure all feature columns are numeric for XGBoost ----------
# Convert object (string) columns to categorical integer codes using the
# union of categories observed in train+test so that mappings are consistent.
obj_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
if obj_cols:
    print(f"Converting object columns to categorical integer codes: {obj_cols}")
    for col in obj_cols:
        # Build combined categories from train+test
        combined = pd.concat([X_train[col].astype(str), X_test[col].astype(str)], ignore_index=True)
        cats = pd.Categorical(combined).categories.tolist()
        mapping = {cat: i for i, cat in enumerate(cats)}
        # Map train and test; unknowns -> -1
        X_train[col] = X_train[col].astype(str).map(mapping).fillna(-1).astype(int)
        X_test[col] = X_test[col].astype(str).map(mapping).fillna(-1).astype(int)
    print("Conversion complete. Example dtypes (train):")
    print(X_train.dtypes.head(20))

# Handle Numeric NaNs by filling with 0 (necessary for clean training)
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)
print("\nNumeric NaNs filled with 0.")


# ==============================
# 2. TRAIN XGBOOST MODEL 
# ==============================
print("\nStarting Model Training...")

N_JOBS = int(os.environ.get("N_JOBS", "4"))

if X_train.shape[0] == 0:
    print("\n⚠️ Training data is empty. Cannot train model.")
    class ZeroModel:
        def predict(self, X):
            return np.zeros(len(X))
    model = ZeroModel()
else:
    # Convert to float32 where safe to reduce memory and speed up training
    try:
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
    except Exception:
        pass

    # Use full dataset for training with a small validation holdout (5%)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=42)
    n_estimators = int(os.environ.get('XGB_N_ESTIMATORS', '1200'))
    max_depth = int(os.environ.get('XGB_MAX_DEPTH', '12'))

    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=float(os.environ.get('XGB_LEARNING_RATE', '0.05')),
        max_depth=max_depth,
        subsample=float(os.environ.get('XGB_SUBSAMPLE', '0.8')),
        colsample_bytree=float(os.environ.get('XGB_COLSAMPLE', '0.8')),
        reg_alpha=float(os.environ.get('XGB_REG_ALPHA', '1')),
        reg_lambda=float(os.environ.get('XGB_REG_LAMBDA', '2')),
        min_child_weight=float(os.environ.get('XGB_MIN_CHILD_WEIGHT', '5')),
        objective="reg:squarederror",
        tree_method=os.environ.get('XGB_TREE_METHOD', 'hist'),
        n_jobs=N_JOBS,
        verbosity=1
    )

    # Fit with early stopping using the small validation set
    try:
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=25, verbose=10)
    except Exception as e:
        print("Primary fit failed (retrying without early stopping):", e)
        model.fit(X_tr, y_tr)

    print("✔ XGBoost Model training complete.")

    # Build a SHAP TreeExplainer for feature importance if available
    try:
        import shap
        try:
            explainer = shap.TreeExplainer(model)
            print("SHAP explainer created.")
        except Exception as e:
            explainer = None
            print("SHAP available but failed to create TreeExplainer.")
    except Exception:
        explainer = None
        print("shap package not available.")

# ==============================
# 3. SAVE AND EXPORT THE MODEL
# ==============================

model_filename = 'xgb_flight_delay_model_trained.joblib'

try:
    # Use joblib.dump() to serialize and save the trained model object ('model')
    joblib.dump(model, model_filename)
    
    print(f"\n✔ Model successfully exported to: {model_filename}")
    print("You can now load this file on any machine using joblib.load().")

except Exception as e:
    print(f"\n❌ Error saving the model: {e}")
    print("Ensure you have the 'joblib' library installed (`pip install joblib`).")
