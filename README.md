# Mobile Price Valuation System

A full-stack Machine Learning web application designed to evaluate and predict mobile phone market prices in **Indian Rupees (INR / ₹)** based on hardware components (RAM, Storage, Battery, Camera Megapixels, CPU Cores, Clock Speed, Display Size, and 5G connectivity). Built with **Python (Flask, scikit-learn)**, **MySQL** database for history persistence, and a clean, warm, light **HTML5/CSS3/JavaScript** frontend.

---

## Key Highlights

- **Direct Indian Rupee (₹) Valuation**: Trained on realistic Indian mobile retail market data distributions with **94.84% $R^2$ accuracy** using a Random Forest Regressor.
- **Indian Market Tier Segments**: Accurately classifies valuations into **Budget (Entry Level)**, **Mid-Range (Value)**, **Upper Mid-Range & Premium**, and **Flagship Tier**.
- **Warm, Clean & Human-Crafted Design**: Professional light theme using warm ivory/beige backgrounds, solid white panels, crisp borders, and zero artificial glowing effects or glassmorphism.
- **MySQL Database Integration**: Persists all calculated valuations in `mobile_price_db.predictions` with timestamps, full component parameters, and market tiers.
- **Dynamic Configuration Profiles**: Quick preset profiles for Budget (~₹9,999), Mid-Range (~₹22,999), and Premium Flagship (~₹74,999).
- **Embedded Fallback Handling**: If MySQL credentials are not yet entered in `.env`, the system automatically activates a local fallback database (`mobile_price_db.sqlite`) so you can immediately test without downtime.

---

## Project Structure

```
mobile_price_prediction/
├── app.py                     # Flask web server & REST API (INR-native)
├── config.py                  # App configuration & DB settings
├── database.py                # MySQL connection & CRUD operations (with fallback)
├── train_model.py             # Dataset generator & Random Forest training pipeline (INR)
├── requirements.txt           # Python package dependencies
├── schema.sql                 # MySQL schema definitions
├── .env.example               # Template for MySQL credentials
├── models/
│   ├── model.pkl              # Saved trained Random Forest model
│   └── scaler.pkl             # Fitted StandardScaler
├── data/
│   └── mobile_price_dataset.csv # 3,500 record training dataset
├── static/
│   ├── css/
│   │   └── style.css          # Clean warm & light responsive stylesheet
│   └── js/
│       └── main.js            # Frontend JavaScript, Fetch API & INR formatting
└── templates/
    └── index.html             # Clean dashboard interface
```

---

## Quick Setup & Execution

### 1. Set Active Workspace
Open your terminal or IDE in the project directory:
```bash
cd C:\Users\Admin\.gemini\antigravity\scratch\mobile_price_prediction
```

### 2. Configure MySQL Database
Copy `.env.example` to `.env`:
```bash
copy .env.example .env
```
Edit `.env` and enter your MySQL root password:
```env
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_mysql_password_here
MYSQL_DATABASE=mobile_price_db
```

### 3. Launch the Application
Start the Flask application:
```bash
python app.py
```

### 4. Open in Browser
Visit:
```
http://localhost:5000
```
- Select a preset profile (Budget, Mid-Range, Flagship) or configure custom hardware parameters.
- Click **"Calculate Price & Save Record"**.
- View the instant valuation in Indian Rupees (₹) and the saved record in the MySQL log table below.
