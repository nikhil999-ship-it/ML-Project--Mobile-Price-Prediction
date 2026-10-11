import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from config import Config
import database

app = Flask(__name__)
app.config.from_object(Config)

# Global variables for model, scaler, and cleaned smartphone catalog
model = None
scaler = None
df_catalog = None

def load_ml_assets():
    global model, scaler
    if os.path.exists(Config.MODEL_PATH) and os.path.exists(Config.SCALER_PATH):
        try:
            model = joblib.load(Config.MODEL_PATH)
            scaler = joblib.load(Config.SCALER_PATH)
            print("ML Model and Scaler loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading ML model/scaler: {e}")
            return False
    else:
        print("Model files not found. Auto-training model...")
        try:
            from train_model import train_and_save_model
            train_and_save_model()
            model = joblib.load(Config.MODEL_PATH)
            scaler = joblib.load(Config.SCALER_PATH)
            return True
        except Exception as e:
            print(f"Failed to auto-train model: {e}")
            return False

def load_catalog_data():
    global df_catalog
    data_dir = getattr(Config, "DATA_DIR", os.path.join(Config.BASE_DIR, "data"))
    catalog_path = os.path.join(data_dir, "cleaned_mobile_data.csv")
    if os.path.exists(catalog_path):
        try:
            df_catalog = pd.read_csv(catalog_path)
            print(f"Loaded {len(df_catalog)} phones into brand catalog matcher.")
            return True
        except Exception as e:
            print(f"Error loading catalog data: {e}")
    return False

# Initialize model, catalog, and database
load_ml_assets()
load_catalog_data()
db_ready, db_status = database.init_db()

@app.route("/")
def home():
    return render_template("landing.html")

@app.route("/calculator")
@app.route("/tool")
def calculator():
    return render_template("index.html")

@app.route("/api/health", methods=["GET"])
def health():
    db_connected, msg = database.test_connection()
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "database_connected": db_connected,
        "database_message": msg
    })

COMPANY_CATALOG = {
    "Samsung": {
        "display_name": "Samsung Galaxy",
        "badge": "Samsung",
        "tagline": "India's Most Trusted Android Brand",
        "origin": "South Korea",
        "models": {
            "Budget": ["Galaxy A14 (~₹13,999)", "Galaxy M14 5G (~₹12,499)"],
            "Mid-Range": ["Galaxy M34 5G (~₹18,999)", "Galaxy A34 5G (~₹24,999)"],
            "Premium": ["Galaxy S21 FE (~₹32,999)", "Galaxy A55 5G (~₹39,999)"],
            "Flagship": ["Galaxy S23 / S24 (~₹69,999+)", "Galaxy S24 Ultra (~₹1,24,999)"]
        }
    },
    "Apple": {
        "display_name": "Apple iPhone",
        "badge": "Apple",
        "tagline": "Premium iOS Ecosystem & Top Resale Value",
        "origin": "United States",
        "models": {
            "Budget": ["iPhone 11 (Certified Used ~₹24,999)", "iPhone 12 (~₹38,999)"],
            "Mid-Range": ["iPhone 13 (~₹45,999)", "iPhone 14 (~₹59,999)"],
            "Premium": ["iPhone 15 (~₹69,999)", "iPhone 15 Plus (~₹79,999)"],
            "Flagship": ["iPhone 16 Pro (~₹1,19,999)", "iPhone 16 Pro Max (~₹1,44,999)"]
        }
    },
    "OnePlus": {
        "display_name": "OnePlus",
        "badge": "OnePlus",
        "tagline": "Fast OxygenOS & Warp Fast Charging",
        "origin": "Never Settle",
        "models": {
            "Budget": ["Nord CE 3 Lite (~₹16,999)"],
            "Mid-Range": ["Nord CE 3 5G (~₹19,999)", "Nord 4 5G (~₹29,999)"],
            "Premium": ["OnePlus 11R (~₹39,999)", "OnePlus 12R (~₹39,999)"],
            "Flagship": ["OnePlus 12 (~₹64,999)"]
        }
    },
    "Xiaomi": {
        "display_name": "Xiaomi / Redmi",
        "badge": "Xiaomi",
        "tagline": "Maximum Hardware Power per Rupee",
        "origin": "Global",
        "models": {
            "Budget": ["Redmi 12 5G (~₹11,999)", "Redmi Note 12 (~₹14,999)"],
            "Mid-Range": ["Redmi Note 13 Pro 5G (~₹24,999)", "Xiaomi 11T Pro (~₹28,999)"],
            "Premium": ["Xiaomi 13 Pro (~₹49,999)"],
            "Flagship": ["Xiaomi 14 Ultra (~₹99,999)"]
        }
    },
    "Realme": {
        "display_name": "Realme",
        "badge": "Realme",
        "tagline": "Youth Trendsetting Styling & SuperVOOC",
        "origin": "Dare to Leap",
        "models": {
            "Budget": ["Realme C55 (~₹11,999)", "Realme 11x 5G (~₹13,999)"],
            "Mid-Range": ["Realme Narzo 60 5G (~₹17,999)", "Realme 12 Pro+ (~₹29,999)"],
            "Premium": ["Realme GT 6T (~₹30,999)", "Realme GT 6 (~₹40,999)"],
            "Flagship": ["Realme GT 5 Pro (~₹54,999)"]
        }
    },
    "Vivo": {
        "display_name": "Vivo",
        "badge": "Vivo",
        "tagline": "Portrait Studio Photography & Aura Light",
        "origin": "Delight Every Moment",
        "models": {
            "Budget": ["Vivo Y27 (~₹13,999)", "Vivo Y36 (~₹16,999)"],
            "Mid-Range": ["Vivo T3 5G (~₹19,999)", "Vivo V29 5G (~₹31,999)"],
            "Premium": ["Vivo V30 Pro (~₹41,999)", "Vivo X90 (~₹49,999)"],
            "Flagship": ["Vivo X100 Pro (~₹89,999)"]
        }
    },
    "Oppo": {
        "display_name": "Oppo",
        "badge": "Oppo",
        "tagline": "ColorOS & Premium Glow Aesthetics",
        "origin": "Inspiration Ahead",
        "models": {
            "Budget": ["Oppo A58 (~₹13,999)", "Oppo A78 5G (~₹18,499)"],
            "Mid-Range": ["Oppo Reno 8 5G (~₹28,499)", "Oppo Reno 11 (~₹34,999)"],
            "Premium": ["Oppo Reno 11 Pro (~₹39,999)"],
            "Flagship": ["Oppo Find N3 Flip (~₹79,999)"]
        }
    },
    "Motorola": {
        "display_name": "Motorola",
        "badge": "Motorola",
        "tagline": "Clean Stock Android & TurboPower Battery",
        "origin": "Hello Moto",
        "models": {
            "Budget": ["Moto G34 5G (~₹10,999)", "Moto G54 5G (~₹15,999)"],
            "Mid-Range": ["Moto G84 5G (~₹18,999)", "Edge 40 Neo (~₹22,999)"],
            "Premium": ["Edge 50 Pro (~₹31,999)", "Edge 50 Ultra (~₹49,999)"],
            "Flagship": ["Motorola Razr 50 Ultra (~₹89,999)"]
        }
    },
    "Google": {
        "display_name": "Google Pixel",
        "badge": "Google",
        "tagline": "Pure Google AI & Computational Photography",
        "origin": "United States",
        "models": {
            "Budget": ["Pixel 6a (Certified Pre-owned ~₹22,999)"],
            "Mid-Range": ["Pixel 7a 5G (~₹39,999)"],
            "Premium": ["Pixel 8a (~₹49,999)", "Pixel 8 (~₹64,999)"],
            "Flagship": ["Pixel 9 Pro XL (~₹1,24,999)"]
        }
    },
    "Nothing": {
        "display_name": "Nothing",
        "badge": "Nothing",
        "tagline": "Iconic Transparent Design & Glyph LEDs",
        "origin": "London, UK",
        "models": {
            "Budget": ["CMF Phone 1 (~₹15,999)"],
            "Mid-Range": ["Nothing Phone (2a) (~₹23,999)", "Nothing Phone 1 (~₹27,999)"],
            "Premium": ["Nothing Phone (2) (~₹36,999)"],
            "Flagship": ["Nothing Phone (3) Upcoming (~₹55,000+)"]
        }
    },
    "Poco": {
        "display_name": "Poco",
        "badge": "Poco",
        "tagline": "Extreme Gaming Processors on a Budget",
        "origin": "Everything You Need",
        "models": {
            "Budget": ["Poco M6 Pro 5G (~₹10,999)", "Poco X5 (~₹14,999)"],
            "Mid-Range": ["Poco X5 Pro (~₹22,999)", "Poco X6 5G (~₹21,999)"],
            "Premium": ["Poco X6 Pro (~₹26,999)", "Poco F6 5G (~₹29,999)"],
            "Flagship": ["Poco F6 Pro (~₹45,999)"]
        }
    },
    "iQOO": {
        "display_name": "iQOO",
        "badge": "iQOO",
        "tagline": "Monster Gaming Engine & Fast Display Rates",
        "origin": "I Quest On and On",
        "models": {
            "Budget": ["iQOO Z9x 5G (~₹12,999)"],
            "Mid-Range": ["iQOO Z9 5G (~₹19,999)", "iQOO Neo 7 (~₹26,999)"],
            "Premium": ["iQOO Neo 9 Pro (~₹34,999)"],
            "Flagship": ["iQOO 12 5G (~₹52,999)"]
        }
    }
}

def predict_company_and_insights(spec_dict, price_inr, tier_name, explicit_company=None):
    global df_catalog
    if price_inr < 15000:
        segment = "Budget"
        top_tier_brands = ["Xiaomi", "Realme", "Motorola", "Samsung"]
    elif price_inr < 30000:
        segment = "Mid-Range"
        top_tier_brands = ["OnePlus", "Samsung", "Vivo", "Poco", "Realme"]
    elif price_inr < 55000:
        segment = "Premium"
        top_tier_brands = ["OnePlus", "Samsung", "Google", "Vivo", "Nothing"]
    else:
        segment = "Flagship"
        top_tier_brands = ["Apple", "Samsung", "Google"]

    matched_key = None
    best_matching_model = None

    # Check explicit company if provided (for API backward-compatibility)
    if explicit_company and str(explicit_company).strip() and str(explicit_company).strip().lower() not in ["any", "any brand", "none"]:
        chosen = str(explicit_company).strip()
        for k in COMPANY_CATALOG:
            if k.lower() == chosen.lower() or chosen.lower() in COMPANY_CATALOG[k]["display_name"].lower():
                matched_key = k
                break

    # Automatically predict company by finding nearest matching phone in dataset
    if not matched_key and df_catalog is not None and not df_catalog.empty:
        try:
            num_cols = [
                'ram_gb', 'storage_gb', 'battery_mah', 'primary_camera_mp',
                'front_camera_mp', 'cpu_speed_ghz', 'cpu_cores', 'screen_size_inch', 'has_5g', 'price_inr'
            ]
            
            # Filter candidates based on real-world mobile platform hardware differences
            cands = df_catalog
            # Real hardware characteristics: Apple iPhones use 6-core chips, <=3500mAh batteries, 6.1" compact screens
            if (
                spec_dict.get('battery_mah', 5000) >= 4000
                or spec_dict.get('screen_size_inch', 6.5) >= 6.4
                or spec_dict.get('cpu_cores', 8) != 6
                or price_inr < 24000
            ):
                cands = cands[cands['brand'] != 'Apple']
            elif price_inr >= 55000 and spec_dict.get('screen_size_inch', 6.5) <= 6.2:
                cands = cands[(cands['brand'] == 'Apple') | (cands['price_inr'] >= 35000)]
            
            if cands.empty:
                cands = df_catalog

            mins = df_catalog[num_cols].min()
            maxs = df_catalog[num_cols].max()
            diffs = (maxs - mins).replace(0, 1)

            target = []
            for c in num_cols:
                val = price_inr if c == 'price_inr' else spec_dict.get(c, 0)
                target.append((val - mins[c]) / diffs[c])
            target = np.array(target)

            norm_cand = (cands[num_cols] - mins) / diffs
            weights = np.array([1.5, 1.2, 1.0, 1.2, 1.0, 1.2, 1.0, 1.0, 0.8, 2.0])
            diff_matrix = (norm_cand.values - target) * weights
            distances = np.linalg.norm(diff_matrix, axis=1)
            best_idx = np.argmin(distances)
            best_row = cands.iloc[best_idx]
            
            best_matching_model = str(best_row['model'])
            raw_brand = str(best_row['brand']).strip()
            for k in COMPANY_CATALOG:
                if k.lower() == raw_brand.lower():
                    matched_key = k
                    break
        except Exception as e:
            print(f"Auto-prediction nearest neighbor error: {e}")

    # Fallback to top segment brand if still not matched
    if not matched_key:
        matched_key = top_tier_brands[0]

    cat = COMPANY_CATALOG[matched_key]
    models = list(cat["models"].get(segment, []))
    if not models:
        for s in ["Mid-Range", "Budget", "Premium", "Flagship"]:
            if cat["models"].get(s):
                models = list(cat["models"][s])
                break

    if not best_matching_model:
        best_matching_model = models[0] if models else f"{cat['display_name']} ({segment})"

    alternatives = [COMPANY_CATALOG[b]["display_name"] for b in top_tier_brands if b != matched_key][:3]

    return {
        "company_name": cat["display_name"],
        "badge_name": cat["badge"],
        "tagline": cat["tagline"],
        "origin": cat["origin"],
        "best_matching_model": best_matching_model,
        "matching_models": models,
        "alternative_companies": alternatives
    }

@app.route("/api/predict", methods=["POST"])
def predict():
    global model, scaler
    if model is None or scaler is None:
        success = load_ml_assets()
        if not success:
            return jsonify({
                "success": False,
                "error": "Machine Learning model is not available. Please run train_model.py first."
            }), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No input data provided."}), 400

        # Required fields extraction and validation
        required_fields = [
            "ram_gb", "storage_gb", "battery_mah", "primary_camera_mp",
            "front_camera_mp", "cpu_speed_ghz", "cpu_cores", "screen_size_inch"
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({"success": False, "error": f"Missing required field: {field}"}), 400

        ram_gb = int(data["ram_gb"])
        storage_gb = int(data["storage_gb"])
        battery_mah = int(data["battery_mah"])
        primary_camera_mp = int(data["primary_camera_mp"])
        front_camera_mp = int(data["front_camera_mp"])
        cpu_speed_ghz = float(data["cpu_speed_ghz"])
        cpu_cores = int(data["cpu_cores"])
        screen_size_inch = float(data["screen_size_inch"])
        has_5g = 1 if data.get("has_5g") in [True, 1, "1", "true"] else 0

        feature_cols = [
            "ram_gb", "storage_gb", "battery_mah", "primary_camera_mp",
            "front_camera_mp", "cpu_speed_ghz", "cpu_cores", "screen_size_inch", "has_5g"
        ]
        features_df = pd.DataFrame([[
            ram_gb, storage_gb, battery_mah, primary_camera_mp,
            front_camera_mp, cpu_speed_ghz, cpu_cores, screen_size_inch, has_5g
        ]], columns=feature_cols)

        scaled_features = scaler.transform(features_df)
        pred_inr = round(float(model.predict(scaled_features)[0]))
        pred_inr = max(pred_inr, 5499)

        # Prepare specification dictionary
        spec_dict = {
            "ram_gb": ram_gb,
            "storage_gb": storage_gb,
            "battery_mah": battery_mah,
            "primary_camera_mp": primary_camera_mp,
            "front_camera_mp": front_camera_mp,
            "cpu_speed_ghz": cpu_speed_ghz,
            "cpu_cores": cpu_cores,
            "screen_size_inch": screen_size_inch,
            "has_5g": has_5g
        }

        # Market Tier categorization (INR)
        from train_model import classify_tier
        tier = classify_tier(pred_inr)

        # Automatically determine/predict the Mobile Company and matching phone models
        company_input = data.get("company_name")
        company_info = predict_company_and_insights(spec_dict, pred_inr, tier, explicit_company=company_input)

        # Calibrate Apple pricing curve if Apple brand is predicted/selected
        if "apple" in company_info["company_name"].lower():
            if pred_inr < 35000:
                pred_inr = max(round(pred_inr * 1.5), 24999)
            elif pred_inr < 55000:
                pred_inr = round(pred_inr * 1.25)
            tier = classify_tier(pred_inr)

        # Save to Database
        spec_dict["company_name"] = company_info["company_name"]
        saved, db_info = database.save_prediction(spec_dict, pred_inr, tier)

        return jsonify({
            "success": True,
            "price_inr": pred_inr,
            "price_tier": tier,
            "company_name": company_info["company_name"],
            "company_badge": company_info["badge_name"],
            "company_tagline": company_info["tagline"],
            "company_origin": company_info["origin"],
            "best_matching_model": company_info["best_matching_model"],
            "matching_models": company_info["matching_models"],
            "alternative_companies": company_info["alternative_companies"],
            "database_saved": saved,
            "database_message": db_info if not saved else "Prediction saved to database."
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/history", methods=["GET"])
def history():
    records = database.get_all_predictions(limit=50)
    return jsonify({"success": True, "records": records})

@app.route("/api/history/<int:record_id>", methods=["DELETE"])
def delete_record(record_id):
    success, msg = database.delete_prediction(record_id)
    return jsonify({"success": success, "message": msg}), (200 if success else 400)

@app.route("/api/init-db", methods=["POST"])
def reinit_db():
    success, msg = database.init_db()
    return jsonify({"success": success, "message": msg})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
