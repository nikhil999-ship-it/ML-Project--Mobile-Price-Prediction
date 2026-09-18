import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def generate_mobile_dataset(n_samples=3500, random_state=42):
    """
    Generates a realistic dataset representing mobile phone specifications and market prices in Indian Rupees (INR).
    Features:
    - ram_gb: RAM in GB (1 to 16)
    - storage_gb: Internal storage in GB (16 to 512)
    - battery_mah: Battery capacity in mAh (1800 to 6000)
    - primary_camera_mp: Main rear camera in Megapixels (5 to 108)
    - front_camera_mp: Selfie camera in Megapixels (2 to 48)
    - cpu_speed_ghz: CPU Clock Speed in GHz (1.2 to 3.4)
    - cpu_cores: Number of CPU Cores (2, 4, 6, 8)
    - screen_size_inch: Screen diagonal in inches (4.7 to 6.9)
    - has_5g: 5G support (0 or 1)
    Target:
    - price_inr: Price in Indian Rupees (₹)
    """
    np.random.seed(random_state)

    ram_options = np.array([2, 3, 4, 6, 8, 12, 16])
    ram_probs = [0.10, 0.15, 0.25, 0.25, 0.15, 0.08, 0.02]
    
    storage_options = np.array([16, 32, 64, 128, 256, 512])
    storage_probs = [0.05, 0.15, 0.30, 0.30, 0.15, 0.05]
    
    cores_options = np.array([2, 4, 6, 8])
    cores_probs = [0.05, 0.25, 0.20, 0.50]

    ram = np.random.choice(ram_options, size=n_samples, p=ram_probs)
    storage = np.random.choice(storage_options, size=n_samples, p=storage_probs)
    cores = np.random.choice(cores_options, size=n_samples, p=cores_probs)
    
    battery = np.random.randint(1800, 6001, size=n_samples)
    cpu_speed = np.round(np.random.uniform(1.2, 3.4, size=n_samples), 2)
    screen_size = np.round(np.random.uniform(4.7, 6.9, size=n_samples), 1)
    
    cam_main_options = np.array([8, 12, 13, 16, 32, 48, 50, 64, 108])
    cam_main = np.random.choice(cam_main_options, size=n_samples)
    cam_front = np.clip(np.round(cam_main * np.random.uniform(0.25, 0.55, size=n_samples)), 2, 48).astype(int)
    
    prob_5g = 0.1 + 0.04 * ram + 0.04 * cores
    prob_5g = np.clip(prob_5g, 0.05, 0.95)
    has_5g = (np.random.rand(n_samples) < prob_5g).astype(int)

    # Realistic Indian Market Valuation (INR)
    base_price = 4200.0
    price = (
        base_price
        + (ram * 2300.0)
        + (storage * 38.0)
        + (battery * 1.85)
        + (cam_main * 115.0)
        + (cam_front * 90.0)
        + (cpu_speed * 3400.0)
        + (cores * 1150.0)
        + (screen_size * 1050.0)
        + (has_5g * 4800.0)
    )

    # Flagship component premium
    flagship_mask = (ram >= 8) & (storage >= 128) & (has_5g == 1) & (cores == 8)
    price[flagship_mask] += np.random.uniform(7000, 16000, size=flagship_mask.sum())

    # Market fluctuation noise
    noise = np.random.normal(0, 1400, size=n_samples)
    price = np.round(np.maximum(price + noise, 5499.0), 0)

    df = pd.DataFrame({
        "ram_gb": ram,
        "storage_gb": storage,
        "battery_mah": battery,
        "primary_camera_mp": cam_main,
        "front_camera_mp": cam_front,
        "cpu_speed_ghz": cpu_speed,
        "cpu_cores": cores,
        "screen_size_inch": screen_size,
        "has_5g": has_5g,
        "price_inr": price
    })

    return df

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
    print("Generating mobile specification dataset in Indian Rupees (INR)...")
    df = generate_mobile_dataset(n_samples=3500)
    
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "mobile_price_dataset.csv")
    df.to_csv(csv_path, index=False)
    print(f"Dataset saved to {csv_path} with {len(df)} records.")

    feature_cols = [
        "ram_gb", "storage_gb", "battery_mah", "primary_camera_mp",
        "front_camera_mp", "cpu_speed_ghz", "cpu_cores", "screen_size_inch", "has_5g"
    ]
    
    X = df[feature_cols]
    y = df["price_inr"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Fitting feature scaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=15,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print("\n--- Model Evaluation (INR) ---")
    print(f"R-squared Score: {r2:.4f} (Accuracy: {r2 * 100:.2f}%)")
    print(f"Mean Absolute Error (MAE): Rs. {mae:.2f}")

    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)

    model_file = os.path.join(models_dir, "model.pkl")
    scaler_file = os.path.join(models_dir, "scaler.pkl")

    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)
    print(f"\nModel saved to: {model_file}")
    print(f"Scaler saved to: {scaler_file}")

    return r2, mae

if __name__ == "__main__":
    train_and_save_model()
