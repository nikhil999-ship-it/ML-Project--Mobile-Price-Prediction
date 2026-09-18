import os
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify
from config import Config
import database

app = Flask(__name__)
app.config.from_object(Config)

# Global variables for model and scaler
model = None
scaler = None

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

# Initialize model and database
load_ml_assets()
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

        import pandas as pd
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

        # Market Tier categorization (INR)
        from train_model import classify_tier
        tier = classify_tier(pred_inr)

        # Save to MySQL Database
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
        
        saved, db_info = database.save_prediction(spec_dict, pred_inr, tier)

        return jsonify({
            "success": True,
            "price_inr": pred_inr,
            "price_tier": tier,
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
