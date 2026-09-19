import os
import re
import pandas as pd
import numpy as np

def clean_mobile_dataset(input_path=None, output_path=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if input_path is None:
        input_path = os.path.join(base_dir, "data", "raw_mobile_data.csv")
    if output_path is None:
        output_path = os.path.join(base_dir, "data", "cleaned_mobile_data.csv")

    print(f"Loading raw dataset from: {input_path}")
    df = pd.read_csv(input_path)
    initial_count = len(df)
    print(f"Total raw rows: {initial_count}")

    # 1. Deduplication
    df = df.drop_duplicates().reset_index(drop=True)
    dedup_count = len(df)
    print(f"Removed {initial_count - dedup_count} exact duplicate rows. Remaining: {dedup_count}")

    # 2. Text Normalization (Brand, Model, OS, Condition)
    # Brand standardization
    def clean_brand(val):
        val = str(val).strip()
        if val.lower() == "oneplus":
            return "OnePlus"
        elif val.lower() == "vivo":
            return "Vivo"
        elif val.lower() == "oppo":
            return "Oppo"
        return val.title()

    df["brand"] = df["brand"].apply(clean_brand)
    df["model"] = df["model"].astype(str).str.strip()
    
    # OS standardization
    def clean_os(val):
        val = str(val).strip().lower()
        if "ios" in val or "apple" in val:
            return "iOS"
        return "Android"
    df["os"] = df["os"].apply(clean_os)

    # Condition standardization
    df["condition"] = df["condition"].astype(str).str.strip().str.capitalize()

    # 3. Numeric Parsing with Regex
    # RAM (GB)
    def parse_ram(val):
        if pd.isna(val) or str(val).strip() == "":
            return np.nan
        match = re.search(r"(\d+)", str(val))
        return int(match.group(1)) if match else np.nan

    df["ram_gb"] = df["ram"].apply(parse_ram)
    median_ram = df["ram_gb"].median()
    def impute_ram(row):
        if pd.isna(row["ram_gb"]):
            if "nord ce 3" in str(row["model"]).lower():
                return 8
            return int(median_ram)
        return int(row["ram_gb"])
    df["ram_gb"] = df.apply(impute_ram, axis=1)

    # Storage (GB)
    def parse_storage(val):
        if pd.isna(val) or str(val).strip() == "":
            return np.nan
        match = re.search(r"(\d+)", str(val))
        return int(match.group(1)) if match else np.nan

    df["storage_gb"] = df["storage"].apply(parse_storage)
    median_storage = df["storage_gb"].median()
    df["storage_gb"] = df["storage_gb"].fillna(median_storage).astype(int)

    # Screen Size (Inches)
    def parse_screen(val):
        if pd.isna(val) or str(val).strip() == "":
            return np.nan
        # Replace comma with dot (e.g. "6,78 inch" -> "6.78")
        s = str(val).replace(",", ".").strip()
        match = re.search(r"(\d+\.?\d*)", s)
        return float(match.group(1)) if match else np.nan

    df["screen_size_inch"] = df["screen_size"].apply(parse_screen)
    median_screen = df["screen_size_inch"].median()
    df["screen_size_inch"] = df["screen_size_inch"].fillna(median_screen).round(2)

    # Camera (MP)
    def parse_camera(val):
        if pd.isna(val) or str(val).strip() == "":
            return np.nan
        match = re.search(r"(\d+)", str(val))
        return int(match.group(1)) if match else np.nan

    df["camera_mp"] = df["camera"].apply(parse_camera)
    median_cam = df["camera_mp"].median()
    def impute_camera(row):
        if pd.isna(row["camera_mp"]):
            if "s21 fe" in str(row["model"]).lower():
                return 12
            return int(median_cam)
        return int(row["camera_mp"])
    df["camera_mp"] = df.apply(impute_camera, axis=1)

    # Battery (mAh)
    def parse_battery(val):
        s = str(val).strip().upper()
        if pd.isna(val) or s == "" or "N/A" in s:
            return np.nan
        match = re.search(r"(\d+)", s)
        return int(match.group(1)) if match else np.nan

    df["battery_mah"] = df["battery"].apply(parse_battery)
    median_bat = df["battery_mah"].median()
    def impute_battery(row):
        if pd.isna(row["battery_mah"]):
            if "moto g54" in str(row["model"]).lower():
                return 6000
            return int(median_bat)
        return int(row["battery_mah"])
    df["battery_mah"] = df.apply(impute_battery, axis=1)

    # 4. Outlier & Error Correction on Price
    # Poco X5 Pro had listed_price '₹22,999' but price '999999' (outlier typo)
    def clean_price_row(row):
        p = float(row["price"])
        # If price is wildly corrupted (> 200,000 for mid-range phone)
        if p > 200000:
            # Check listed_price
            lp_match = re.search(r"[\d,]+", str(row["listed_price"]))
            if lp_match:
                clean_lp = int(lp_match.group(0).replace(",", ""))
                print(f"Corrected corrupted price typo for {row['model']}: {p} -> {clean_lp}")
                return clean_lp
        return p

    df["price_inr"] = df.apply(clean_price_row, axis=1).astype(int)

    # 5. Model-specific hardware specifications (Front camera, CPU, 5G)
    specs_map = {
        "Samsung Galaxy A14": {"front_camera_mp": 13, "cpu_speed_ghz": 2.0, "cpu_cores": 8, "has_5g": 0},
        "Redmi Note 12": {"front_camera_mp": 13, "cpu_speed_ghz": 2.0, "cpu_cores": 8, "has_5g": 1},
        "iPhone 12": {"front_camera_mp": 12, "cpu_speed_ghz": 3.1, "cpu_cores": 6, "has_5g": 1},
        "OnePlus Nord CE 3": {"front_camera_mp": 16, "cpu_speed_ghz": 2.7, "cpu_cores": 8, "has_5g": 1},
        "Realme Narzo 60": {"front_camera_mp": 16, "cpu_speed_ghz": 2.2, "cpu_cores": 8, "has_5g": 1},
        "Vivo V29": {"front_camera_mp": 50, "cpu_speed_ghz": 2.4, "cpu_cores": 8, "has_5g": 1},
        "Oppo A78": {"front_camera_mp": 8, "cpu_speed_ghz": 2.4, "cpu_cores": 8, "has_5g": 0},
        "Samsung S21 FE": {"front_camera_mp": 32, "cpu_speed_ghz": 2.9, "cpu_cores": 8, "has_5g": 1},
        "iPhone 13": {"front_camera_mp": 12, "cpu_speed_ghz": 3.2, "cpu_cores": 6, "has_5g": 1},
        "Pixel 7a": {"front_camera_mp": 13, "cpu_speed_ghz": 2.85, "cpu_cores": 8, "has_5g": 1},
        "Nothing Phone 1": {"front_camera_mp": 16, "cpu_speed_ghz": 2.5, "cpu_cores": 8, "has_5g": 1},
        "Moto G54": {"front_camera_mp": 16, "cpu_speed_ghz": 2.2, "cpu_cores": 8, "has_5g": 1},
        "Poco X5 Pro": {"front_camera_mp": 16, "cpu_speed_ghz": 2.4, "cpu_cores": 8, "has_5g": 1},
        "iPhone 11": {"front_camera_mp": 12, "cpu_speed_ghz": 2.65, "cpu_cores": 6, "has_5g": 0},
        "Realme C55": {"front_camera_mp": 8, "cpu_speed_ghz": 2.0, "cpu_cores": 8, "has_5g": 0},
        "Vivo Y36": {"front_camera_mp": 16, "cpu_speed_ghz": 2.4, "cpu_cores": 8, "has_5g": 0},
        "Oppo Reno 8": {"front_camera_mp": 32, "cpu_speed_ghz": 3.0, "cpu_cores": 8, "has_5g": 1},
        "Samsung M34": {"front_camera_mp": 13, "cpu_speed_ghz": 2.4, "cpu_cores": 8, "has_5g": 1},
        "OnePlus 11R": {"front_camera_mp": 16, "cpu_speed_ghz": 3.2, "cpu_cores": 8, "has_5g": 1},
        "iPhone 14": {"front_camera_mp": 12, "cpu_speed_ghz": 3.2, "cpu_cores": 6, "has_5g": 1},
    }

    df["primary_camera_mp"] = df["camera_mp"]
    df["front_camera_mp"] = df["model"].apply(lambda m: specs_map.get(m, {}).get("front_camera_mp", 16))
    df["cpu_speed_ghz"] = df["model"].apply(lambda m: specs_map.get(m, {}).get("cpu_speed_ghz", 2.4))
    df["cpu_cores"] = df["model"].apply(lambda m: specs_map.get(m, {}).get("cpu_cores", 8))
    df["has_5g"] = df["model"].apply(lambda m: specs_map.get(m, {}).get("has_5g", 1))

    # 6. Clean Final Columns
    cleaned_columns = [
        "model", "brand", "os", "ram_gb", "storage_gb", 
        "screen_size_inch", "primary_camera_mp", "front_camera_mp",
        "cpu_speed_ghz", "cpu_cores", "battery_mah", "has_5g",
        "condition", "price_inr"
    ]
    df_clean = df[cleaned_columns]

    # Save Cleaned CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(f"\nCleaned dataset successfully saved to: {output_path}")
    print("\n--- Cleaned Dataset Sample ---")
    print(df_clean.head(10).to_string(index=False))

    return df_clean

if __name__ == "__main__":
    clean_mobile_dataset()
