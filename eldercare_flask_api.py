"""
ElderCare+ Flask API with MongoDB Atlas
========================================
REST API with full database connectivity

ENDPOINTS:
  GET  /health          → Check server
  POST /predict         → Get risk level
  POST /analyze         → Risk + Gemini recommendation
  POST /save_reading    → Save health reading to MongoDB
  GET  /history/<user>  → Get patient history from MongoDB

HOW TO RUN:
  cd C:/Users/AJAY/OneDrive/Desktop/eldercare
  python eldercare_flask_api.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ─────────────────────────────────────────────
# ⚠️  UPDATE THESE VALUES
# ─────────────────────────────────────────────
import os
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyDlS0FokFFaAdjDnoNbpUaB-ZW3Hx-uVak')

from urllib.parse import quote_plus
password  = quote_plus("eldercare@123")
MONGO_URI = f"mongodb+srv://eldercare_user:{password}@cluster0.fxmuvwk.mongodb.net/?appName=Cluster0"
# Replace YOUR_PASSWORD with your actual MongoDB password

# ─────────────────────────────────────────────
# FLASK SETUP
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

print("=" * 55)
print("   ElderCare+ Flask API with MongoDB")
print("=" * 55)

# ─────────────────────────────────────────────
# MONGODB CONNECTION
# ─────────────────────────────────────────────
DB_CLIENT     = None
DB            = None
READINGS_COL  = None
USERS_COL     = None

try:
    DB_CLIENT    = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    DB_CLIENT.admin.command('ping')
    DB           = DB_CLIENT['eldercare_db']
    READINGS_COL = DB['health_readings']
    USERS_COL    = DB['users']
    print("✅ MongoDB Atlas connected!")
    print(f"   Database: eldercare_db")
    print(f"   Collections: health_readings, users")
except Exception as e:
    print(f"⚠️  MongoDB connection failed: {e}")
    print("   App will work but data won't be saved to database")

# ─────────────────────────────────────────────
# ML MODEL
# ─────────────────────────────────────────────
def load_or_train_model():
    MODEL_FILE = 'eldercare_model.pkl'
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, 'rb') as f:
                model, feature_cols = pickle.load(f)
            print("✅ ML Model loaded from file")
            return model, feature_cols
        except Exception:
            os.remove(MODEL_FILE)

    print("⚙️  Training fresh ML model...")
    df = pd.read_csv('preprocessed_dataset.csv').fillna(0)
    X  = df.drop(columns=['risk_numeric'])
    y  = df['risk_numeric']
    feature_cols = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"✅ Model trained — Accuracy: {acc*100:.2f}%")
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump((model, feature_cols), f)
    return model, feature_cols

try:
    ML_MODEL, FEATURE_COLS = load_or_train_model()
    MODEL_READY = True
except Exception as e:
    print(f"⚠️  Model failed: {e}")
    MODEL_READY = False

# Gemini setup
GEMINI_CLIENT = None
if GEMINI_AVAILABLE and GEMINI_API_KEY != "AIzaSyDlS0FokFFaAdjDnoNbpUaB-ZW3Hx-uVak":
    try:
        GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
        print("✅ Gemini AI connected")
    except Exception as e:
        print(f"⚠️  Gemini failed: {e}")
else:
    print("⚠️  Gemini not configured")

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def prepare_features(data):
    age    = float(data.get('age', 70))
    bmi    = float(data.get('bmi', 26))
    bp_sys = float(data.get('bp_systolic', 120))
    bp_dia = float(data.get('bp_diastolic', 80))
    sugar  = float(data.get('blood_sugar', 100))
    hr     = float(data.get('heart_rate', 75))
    spo2   = float(data.get('spo2', 97))
    gender = data.get('gender', 'Female')
    cond   = data.get('existing_condition', 'None')

    def norm(v, mn, mx): return (v - mn) / (mx - mn)

    row = {
        'gender_encoded'    : 1 if gender == 'Male' else 0,
        'cond_Diabetes'     : 1 if cond == 'Diabetes' else 0,
        'cond_Heart Disease': 1 if cond == 'Heart Disease' else 0,
        'cond_Hypertension' : 1 if cond == 'Hypertension' else 0,
        'cond_None'         : 1 if cond == 'None' else 0,
        'age': age, 'bmi': bmi, 'bp_systolic': bp_sys,
        'bp_diastolic': bp_dia, 'blood_sugar': sugar,
        'heart_rate': hr, 'spo2': spo2,
        'age_norm'         : norm(age,    60,  90),
        'bmi_norm'         : norm(bmi,    16,  42),
        'bp_systolic_norm' : norm(bp_sys, 90,  210),
        'bp_diastolic_norm': norm(bp_dia, 50,  130),
        'blood_sugar_norm' : norm(sugar,  65,  350),
        'heart_rate_norm'  : norm(hr,     45,  150),
        'spo2_norm'        : norm(spo2,   88,  100),
    }
    df_row = pd.DataFrame([row])
    for col in FEATURE_COLS:
        if col not in df_row.columns:
            df_row[col] = 0
    return df_row[FEATURE_COLS]


def get_gemini_recommendation(data, risk_label):
    if GEMINI_CLIENT is None:
        return "Gemini not configured.\nHINDI: Gemini configure nahi hai."
    try:
        prompt = f"""
You are a caring healthcare assistant for elderly Indian patients.
Give recommendation in BOTH English and Hindi.
Patient: Age {data.get('age')}, {data.get('gender')}, {data.get('existing_condition')}
BP: {data.get('bp_systolic')}/{data.get('bp_diastolic')}, Sugar: {data.get('blood_sugar')},
HR: {data.get('heart_rate')}, SpO2: {data.get('spo2')}, Risk: {risk_label}
FORMAT:
ENGLISH: [4-5 sentences]
HINDI: [same in Hindi]
DOCTOR VISIT: [Urgent / Within a week / Routine]
"""
        response = GEMINI_CLIENT.models.generate_content(
            model="gemini-2.0-flash-lite", contents=prompt)
        return response.text.strip()
    except Exception as e:
        return f"Gemini Error: {str(e)}"


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

# 1. Health Check
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status'   : 'running',
        'message'  : 'ElderCare+ API is active!',
        'ml_model' : 'ready' if MODEL_READY else 'not loaded',
        'gemini'   : 'connected' if GEMINI_CLIENT else 'not configured',
        'mongodb'  : 'connected' if DB_CLIENT else 'not connected',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })


# 2. Predict Risk
@app.route('/predict', methods=['POST'])
def predict_risk():
    if not MODEL_READY:
        return jsonify({'error': 'ML model not loaded'}), 500
    try:
        data       = request.get_json()
        features   = prepare_features(data)
        prediction = ML_MODEL.predict(features)[0]
        probas     = ML_MODEL.predict_proba(features)[0]
        label_map  = {0: 'Low', 1: 'Medium', 2: 'High'}
        return jsonify({
            'risk_label'   : label_map[int(prediction)],
            'risk_numeric' : int(prediction),
            'confidence'   : round(float(max(probas)), 4),
            'probabilities': {
                'Low'   : round(float(probas[0]), 4),
                'Medium': round(float(probas[1]), 4),
                'High'  : round(float(probas[2]), 4)
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# 3. Full Analysis (Predict + Gemini + Save to MongoDB)
@app.route('/analyze', methods=['POST'])
def analyze():
    if not MODEL_READY:
        return jsonify({'error': 'ML model not loaded'}), 500
    try:
        data       = request.get_json()
        features   = prepare_features(data)
        prediction = ML_MODEL.predict(features)[0]
        probas     = ML_MODEL.predict_proba(features)[0]
        label_map  = {0: 'Low', 1: 'Medium', 2: 'High'}
        risk_label = label_map[int(prediction)]
        rec        = get_gemini_recommendation(data, risk_label)

        # Save to MongoDB
        if READINGS_COL is not None:
            reading_doc = {
                'user_name'         : data.get('user_name', 'Unknown'),
                'age'               : data.get('age'),
                'gender'            : data.get('gender'),
                'existing_condition': data.get('existing_condition'),
                'bp_systolic'       : data.get('bp_systolic'),
                'bp_diastolic'      : data.get('bp_diastolic'),
                'blood_sugar'       : data.get('blood_sugar'),
                'heart_rate'        : data.get('heart_rate'),
                'spo2'              : data.get('spo2'),
                'bmi'               : data.get('bmi'),
                'risk_label'        : risk_label,
                'risk_numeric'      : int(prediction),
                'confidence'        : round(float(max(probas)), 4),
                'recommendation'    : rec,
                'timestamp'         : datetime.now()
            }
            READINGS_COL.insert_one(reading_doc)

        return jsonify({
            'status'        : 'success',
            'risk_label'    : risk_label,
            'risk_numeric'  : int(prediction),
            'confidence'    : round(float(max(probas)), 4),
            'probabilities' : {
                'Low'   : round(float(probas[0]), 4),
                'Medium': round(float(probas[1]), 4),
                'High'  : round(float(probas[2]), 4)
            },
            'recommendation': rec,
            'saved_to_db'   : READINGS_COL is not None,
            'timestamp'     : datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# 4. Get History from MongoDB
@app.route('/history/<user_name>', methods=['GET'])
def get_history(user_name):
    if READINGS_COL is None:
        return jsonify({'error': 'MongoDB not connected'}), 500
    try:
        readings = list(READINGS_COL.find(
            {'user_name': user_name},
            {'_id': 0}
        ).sort('timestamp', -1).limit(10))

        # Convert datetime to string
        for r in readings:
            if 'timestamp' in r:
                r['timestamp'] = str(r['timestamp'])[:16]

        return jsonify({
            'user_name': user_name,
            'count'    : len(readings),
            'readings' : readings
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# 5. Save User Profile
@app.route('/save_user', methods=['POST'])
def save_user():
    if USERS_COL is None:
        return jsonify({'error': 'MongoDB not connected'}), 500
    try:
        data = request.get_json()
        USERS_COL.update_one(
            {'user_name': data.get('user_name')},
            {'$set': {**data, 'updated_at': datetime.now()}},
            upsert=True
        )
        return jsonify({'status': 'success', 'message': 'User saved!'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("\n✅ Endpoints ready:")
    print("   GET  /health")
    print("   POST /predict")
    print("   POST /analyze      ← Flutter uses this")
    print("   GET  /history/<name>")
    print("   POST /save_user")
    print("\n   Open: http://localhost:5000/health")
    print("   Press CTRL+C to stop\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
