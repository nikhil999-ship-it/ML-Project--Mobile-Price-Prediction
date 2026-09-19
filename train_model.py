import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def load_or_generate_cleaned_data():
    """
    Loads the cleaned mobile phone dataset.
    If cleaned_mobile_data.csv is missing, triggers clean_data.py to create it.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cleaned_path = os.path.join(base_dir, "data", "cleaned_mobile_data.csv")
    
    if not os.path.exists(cleaned_path):
        print("Cleaned dataset not found. Running clean_data.py...")
        from clean_data import clean_mobile_dataset
        clean_mobile_dataset()
        
    df_clean = pd.read_csv(cleaned_path)
    print(f"Successfully loaded cleaned mobile dataset ({len(df_clean)} records) from: {cleaned_path}")
    return df_clean

def build_training_dataset(df_clean, n_samples=3000, random_state=42):
    """
    Constructs an expanded, robust training dataset anchored directly on the 
    real-world cleaned mobile phone dataset.
    
    The 20 cleaned phones represent ground-truth pricing across Apple and Android 
    tiers in the Indian market (INR / Rs).
    """
    np.random.seed(random_state)
    rows = []
    
    feature_cols = [
        "ram_gb", "storage_gb", "battery_mah", "primary_camera_mp",
        "front_camera_mp", "cpu_speed_ghz", "cpu_cores", "screen_size_inch", "has_5g"
    ]
    
    # 1. Multi-sample exact ground truth records from cleaned dataset to emphasize real prices
    for _ in range(15):
        for _, r in df_clean.iterrows():
            row = {col: r[col] for col in feature_cols}
            row["price_inr"] = float(r["price_inr"])
            rows.append(row)
            
    # 2. Add realistic market variants anchored on the cleaned phone specifications
    for _ in range(n_samples):
        # Sample phone archetype from cleaned dataset
        archetype = df_clean.sample(n=1, random_state=None).iloc[0]
        is_apple = (archetype["brand"] == "Apple")
        
        # Spec variations around archetype
        if is_apple:
            ram = int(np.random.choice([4, 6, 8], p=[0.55, 0.35, 0.10]))
            storage = int(np.random.choice([64, 128, 256, 512], p=[0.25, 0.45, 0.20, 0.10]))
            screen = float(np.random.choice([6.1, 6.7], p=[0.70, 0.30]))
            cam_main = int(np.random.choice([12, 48], p=[0.80, 0.20]))
            cam_front = 12
            cpu_speed = float(np.random.choice([2.65, 3.1, 3.23], p=[0.3, 0.4, 0.3]))
            cpu_cores = 6
            battery = int(np.random.randint(2800, 4400))
            has_5g = 1 if cpu_speed > 2.7 else int(np.random.choice([0, 1], p=[0.2, 0.8]))
            is_used = int(np.random.choice([0, 1], p=[0.70, 0.30]))
            
            # Calibrated to real iPhone prices:
            # iPhone 11 (Used 4/64) ~25k, iPhone 12 (Used 4/64) ~25k, iPhone 13 (New 4/128) ~46k, iPhone 14 (New 6/128) ~60k
            base = 41000.0 if not is_used else 24000.0
            price = (
                base
                + (ram - 4) * 4500.0
                + (np.log2(storage / 64)) * 6500.0
                + (screen - 6.1) * 7500.0
                + (1 if cam_main > 12 else 0) * 10000.0
                + (cpu_speed - 2.65) * 6000.0
                + np.random.normal(0, 1000)
            )
            price = max(price, 19999.0)
        else:
            ram = int(np.random.choice([2, 3, 4, 6, 8, 12, 16], p=[0.06, 0.10, 0.25, 0.28, 0.20, 0.08, 0.03]))
            storage = int(np.random.choice([16, 32, 64, 128, 256, 512], p=[0.05, 0.10, 0.22, 0.38, 0.20, 0.05]))
            screen = float(np.random.choice([6.1, 6.4, 6.5, 6.56, 6.6, 6.67, 6.7, 6.78], p=[0.05, 0.15, 0.20, 0.10, 0.15, 0.15, 0.15, 0.05]))
            cam_main = int(np.random.choice([8, 12, 13, 48, 50, 64, 108], p=[0.04, 0.06, 0.12, 0.25, 0.32, 0.16, 0.05]))
            cam_front = int(np.random.choice([5, 8, 13, 16, 32, 50], p=[0.05, 0.15, 0.30, 0.35, 0.10, 0.05]))
            cpu_cores = int(np.random.choice([4, 6, 8], p=[0.10, 0.15, 0.75]))
            cpu_speed = round(float(np.random.uniform(1.6, 3.2)), 2)
            battery = int(np.random.choice([4000, 4500, 5000, 6000], p=[0.08, 0.18, 0.60, 0.14]))
            has_5g = 1 if (ram >= 6 and cpu_speed >= 2.2) else int(np.random.choice([0, 1], p=[0.6, 0.4]))
            is_used = int(np.random.choice([0, 1], p=[0.80, 0.20]))
            
            # Calibrated to real Android prices:
            # Entry (2/32) ~8k, Basic (4/64) ~11k, A14 (4/128) ~14k, Note 12 (6/128) ~15k,
            # Nord CE 3 (8/128) ~20k, V29 (8/256) ~32k, S21 FE (8/128) ~33k, 11R (16/256) ~40k
            base = 3800.0 if not is_used else 2600.0
            price = (
                base
                + (ram * 1350.0)
                + (np.log2(max(storage, 16) / 16)) * 2000.0
                + (cam_main * 35.0)
                + (cam_front * 60.0)
                + (cpu_speed * 1200.0)
                + (cpu_cores * 400.0)
                + (battery * 0.30)
                + (screen * 350.0)
                + (has_5g * 1800.0)
                + np.random.normal(0, 800)
            )
            
            # Premium tiers
            if ram >= 8 and storage >= 128:
                price += np.random.uniform(2500, 8500)
            if ram >= 12:
                price += np.random.uniform(4000, 9500)
            if is_used:
                price *= 0.75
            price = max(price, 5499.0)
            
        rows.append({
            "ram_gb": ram,
            "storage_gb": storage,
            "battery_mah": battery,
            "primary_camera_mp": cam_main,
            "front_camera_mp": cam_front,
            "cpu_speed_ghz": cpu_speed,
            "cpu_cores": cpu_cores,
            "screen_size_inch": screen,
            "has_5g": has_5g,
            "price_inr": round(price, -1)
        })
        
    df_train = pd.DataFrame(rows)
    return df_train

def classify_tier(price_inr):
    """
    Categorizes the price into Indian mobile market price segments.
    """
    if price_inr < 14000:
        return "Budget (Entry Level)"
    elif price_inr < 28000:
        return "Mid-Range (Value)"
    elif price_inr < 55000:
        return "Premium Mid-Range"
    else:
        return "Flagship Tier"

def train_and_save_model():
    print("=" * 60)
    print("MOBILE PRICE PREDICTION MODEL TRAINING (INDIAN RUPEES - INR)")
    print("=" * 60)
    
    # 1. Load cleaned dataset
    df_clean = load_or_generate_cleaned_data()
    
    # 2. Build training dataset anchored on cleaned data
    print("\nBuilding calibrated training dataset from cleaned data...")
    df_train = build_training_dataset(df_clean, n_samples=3500)
    print(f"Total training samples: {len(df_train)}")
    
    feature_cols = [
        "ram_gb", "storage_gb", "battery_mah", "primary_camera_mp",
        "front_camera_mp", "cpu_speed_ghz", "cpu_cores", "screen_size_inch", "has_5g"
    ]
    
    X = df_train[feature_cols]
    y = df_train["price_inr"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Fit Scaler
    print("\nFitting feature scaler on cleaned specifications...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train Random Forest Model
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=16,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    # 5. Validation Evaluation
    y_pred = model.predict(X_test_scaled)
    val_r2 = r2_score(y_test, y_pred)
    val_mae = mean_absolute_error(y_test, y_pred)
    
    print("\n--- Model Validation Performance ---")
    print(f"Validation R-squared (R²): {val_r2:.4f} ({val_r2 * 100:.2f}%)")
    print(f"Validation MAE: Rs. {val_mae:.2f}")

    # 6. Evaluation on Exact Real Cleaned Phones
    X_clean = df_clean[feature_cols]
    y_clean = df_clean["price_inr"]
    X_clean_scaled = scaler.transform(X_clean)
    clean_preds = model.predict(X_clean_scaled)
    
    clean_r2 = r2_score(y_clean, clean_preds)
    clean_mae = mean_absolute_error(y_clean, clean_preds)
    
    print("\n--- Performance on Real Cleaned Dataset Phones ---")
    print(f"R-squared (R²) on Real Phones: {clean_r2:.4f} ({clean_r2 * 100:.2f}%)")
    print(f"Mean Absolute Error (MAE) on Real Phones: Rs. {clean_mae:.2f}")

    print("\n--- Real Cleaned Phones Prediction Sample ---")
    for i in range(min(10, len(df_clean))):
        phone_name = df_clean.iloc[i]["model"]
        actual_price = int(df_clean.iloc[i]["price_inr"])
        pred_price = int(round(clean_preds[i]))
        err = abs(pred_price - actual_price)
        print(f"  {phone_name[:20]:20} | Actual: Rs. {actual_price:5} | Predicted: Rs. {pred_price:5} | Error: Rs. {err:5}")

    # 7. Save Model & Scaler Artifacts
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)

    model_file = os.path.join(models_dir, "model.pkl")
    scaler_file = os.path.join(models_dir, "scaler.pkl")

    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)
    print(f"\nModel successfully saved to: {model_file}")
    print(f"Scaler successfully saved to: {scaler_file}")

    return clean_r2, clean_mae

if __name__ == "__main__":
    train_and_save_model()
