"""
MediShield Pro - Flask API Backend
Serves ML predictions for health insurance cost estimation.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import json
import os

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)  # Enable CORS for frontend

# ============================================
# Database Configuration
# ============================================
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///medishield.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Database Model
class Inquiry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100))
    mobile = db.Column(db.String(15))
    email = db.Column(db.String(100))
    pincode = db.Column(db.String(10))
    location = db.Column(db.String(100))
    gender = db.Column(db.String(10))
    members_count = db.Column(db.Integer)
    members_json = db.Column(db.Text)  # Stores detailed member info as JSON string
    conditions_json = db.Column(db.Text)
    is_smoker = db.Column(db.Boolean)
    bmi = db.Column(db.Float)
    annual_income = db.Column(db.Float)
    sum_insured = db.Column(db.Integer)
    tenure = db.Column(db.Integer)
    predicted_premium = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'full_name': self.full_name,
            'mobile': self.mobile,
            'email': self.email,
            'pincode': self.pincode,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S')
        }

# Create database tables
with app.app_context():
    db.create_all()

# ============================================
# OTP Memory Store (For Demo Security)
# ============================================
import random
TEMP_OTPS = {}

@app.route('/api/send-otp', methods=['POST'])
def send_otp():
    """Generate and store a random OTP."""
    data = request.get_json()
    mobile = data.get('mobile')
    if not mobile or len(mobile) < 10:
        return jsonify({'success': False, 'message': 'Invalid mobile number'}), 400
    
    otp = str(random.randint(1000, 9999))
    TEMP_OTPS[mobile] = otp
    print(f"🔑 [SECURITY] OTP for {mobile}: {otp}")
    
    return jsonify({
        'success': True, 
        'message': 'OTP sent successfully',
        'otp_debug': otp # In a real app, this would NOT be in the response
    })

@app.route('/api/verify-otp', methods=['POST'])
def verify_otp():
    """Verify the provided OTP."""
    data = request.get_json()
    mobile = data.get('mobile')
    otp = data.get('otp')
    
    if mobile in TEMP_OTPS and TEMP_OTPS[mobile] == otp:
        # OPTIONAL: Clear OTP after successful verification
        # del TEMP_OTPS[mobile] 
        return jsonify({'success': True, 'message': 'Verification successful'})
    
    return jsonify({'success': False, 'message': 'Invalid verification code'}), 401

# ============================================
# Load Model & Artifacts
# ============================================
MODEL = None
SCALER = None
FEATURE_COLS = None
METRICS = None

def load_model():
    """Load the trained model and preprocessing artifacts."""
    global MODEL, SCALER, FEATURE_COLS, METRICS
    
    model_path = 'models/xgb_insurance_model.pkl'
    scaler_path = 'models/scaler.pkl'
    features_path = 'models/feature_cols.json'
    metrics_path = 'models/metrics.json'
    
    if not os.path.exists(model_path):
        print("Model not found. Please run 'python train_model.py' first.")
        return False
    
    MODEL = joblib.load(model_path)
    SCALER = joblib.load(scaler_path)
    
    with open(features_path, 'r') as f:
        FEATURE_COLS = json.load(f)
    
    with open(metrics_path, 'r') as f:
        METRICS = json.load(f)
    
    print("Model loaded successfully!")
    print(f"   R² Score: {METRICS['test_r2']:.4f}")
    print(f"   MAE: Rs.{METRICS['test_mae']:,.0f}")
    return True


# ============================================
# Feature Engineering (must match training)
# ============================================
def engineer_features_single(data):
    """
    Apply the same feature engineering used during training.
    data is a dict with raw input features.
    """
    age = data.get('age', 30)
    bmi = data.get('bmi', 24.0)
    
    # Age group
    if age <= 17: age_group = 0
    elif age <= 25: age_group = 1
    elif age <= 35: age_group = 2
    elif age <= 45: age_group = 3
    elif age <= 55: age_group = 4
    elif age <= 65: age_group = 5
    else: age_group = 6
    
    # BMI category
    if bmi < 18.5: bmi_category = 0
    elif bmi < 25: bmi_category = 1
    elif bmi < 30: bmi_category = 2
    elif bmi < 35: bmi_category = 3
    else: bmi_category = 4
    
    diabetes = data.get('diabetes', 0)
    hypertension = data.get('hypertension', 0)
    heart_disease = data.get('heart_disease', 0)
    thyroid = data.get('thyroid', 0)
    asthma = data.get('asthma', 0)
    smoker = data.get('smoker', 0)
    num_members = data.get('num_members', 1)
    sum_insured = data.get('sum_insured', 10)
    
    total_conditions = diabetes + hypertension + heart_disease + thyroid + asthma
    has_condition = 1 if total_conditions > 0 else 0
    age_smoker = age * smoker
    bmi_conditions = bmi * total_conditions
    age_conditions = age * total_conditions
    sum_per_member = sum_insured / max(num_members, 1)
    risk_score = (age / 80 * 0.3 +
                  bmi / 50 * 0.15 +
                  smoker * 0.2 +
                  total_conditions / 5 * 0.25 +
                  num_members / 6 * 0.1)
    
    features = {
        'age': age,
        'gender': data.get('gender', 0),
        'bmi': bmi,
        'num_children': data.get('num_children', 0),
        'smoker': smoker,
        'region': data.get('region', 0),
        'diabetes': diabetes,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'thyroid': thyroid,
        'asthma': asthma,
        'num_members': num_members,
        'sum_insured': sum_insured,
        'tenure': data.get('tenure', 1),
        'city_tier': data.get('city_tier', 2),
        'avg_family_age': data.get('avg_family_age', age),
        'age_group': age_group,
        'bmi_category': bmi_category,
        'total_conditions': total_conditions,
        'has_condition': has_condition,
        'age_smoker': age_smoker,
        'bmi_conditions': bmi_conditions,
        'age_conditions': age_conditions,
        'sum_per_member': sum_per_member,
        'risk_score': risk_score,
    }
    
    return features


# ============================================
# Routes
# ============================================

@app.route('/')
def index():
    """Serve the main HTML page."""
    return send_from_directory('.', 'index.html')


@app.route('/api/health')
def health_check():
    """API health check."""
    return jsonify({
        'status': 'ok',
        'model_loaded': MODEL is not None,
        'metrics': METRICS,
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict insurance premium for given input.
    
    Expected JSON body:
    {
        "members": [
            {"type": "self", "age": 30, "label": "Self"},
            {"type": "wife", "age": 28, "label": "Wife"},
            ...
        ],
        "conditions": ["diabetes", "hypertension"],
        "smoker": false,
        "bmi": 24.5,
        "sum_insured": 10,
        "tenure": 1,
        "pincode": "533221",
        "gender": "male",
        "num_children": 1
    }
    
    Returns:
    {
        "success": true,
        "plans": [
            {
                "name": "...",
                "premium": 27512,
                "original_price": 37442,
                "discount": 27,
                "features": [...],
                "description": "..."
            },
            ...
        ],
        "prediction_details": {...}
    }
    """
    if MODEL is None:
        return jsonify({'success': False, 'error': 'Model not loaded. Run train_model.py first.'}), 500
    
    try:
        data = request.get_json()
        
        members = data.get('members', [{'type': 'self', 'age': 30}])
        conditions = data.get('conditions', [])
        is_smoker = 1 if data.get('smoker', False) else 0
        bmi = data.get('bmi', 24.0) or 24.0
        sum_insured = data.get('sum_insured', 10)
        tenure = data.get('tenure', 1)
        pincode = data.get('pincode', '')
        gender_str = data.get('gender', 'male')
        num_children = data.get('num_children', 0)
        
        # Determine region and city tier from pincode
        region, city_tier = get_region_from_pincode(pincode)
        
        # Gender encoding
        gender = 0 if gender_str == 'male' else 1
        
        # Condition flags
        diabetes = 1 if 'diabetes' in conditions else 0
        hypertension = 1 if 'hypertension' in conditions else 0
        heart_disease = 1 if 'heart' in conditions else 0
        thyroid = 1 if 'thyroid' in conditions else 0
        asthma = 1 if 'asthma' in conditions else 0
        
        # Count children from members
        child_members = [m for m in members if m.get('type') in ('son', 'daughter')]
        num_children = len(child_members)
        
        # Calculate average family age
        ages = [m.get('age', 30) for m in members]
        avg_family_age = sum(ages) / len(ages) if ages else 30
        
        # Get the primary member's age (oldest adult or self)
        primary_age = ages[0] if ages else 30
        
        # Predict for each member and sum up
        total_premium = 0
        member_premiums = []
        
        for member in members:
            member_age = member.get('age', 30)
            
            # Build feature dict for this member
            raw_data = {
                'age': member_age,
                'gender': gender,
                'bmi': float(bmi),
                'num_children': num_children,
                'smoker': is_smoker,
                'region': region,
                'diabetes': diabetes,
                'hypertension': hypertension,
                'heart_disease': heart_disease,
                'thyroid': thyroid,
                'asthma': asthma,
                'num_members': len(members),
                'sum_insured': sum_insured,
                'tenure': tenure,
                'city_tier': city_tier,
                'avg_family_age': avg_family_age,
            }
            
            # Engineer features
            features = engineer_features_single(raw_data)
            
            # Create DataFrame with correct column order
            feature_df = pd.DataFrame([features])[FEATURE_COLS]
            
            # Scale
            feature_scaled = SCALER.transform(feature_df.values)
            
            # Predict
            predicted = MODEL.predict(feature_scaled)[0]
            predicted = max(2000, float(predicted))
            
            member_premiums.append({
                'member': member.get('label', member.get('type', 'Self')),
                'age': member_age,
                'individual_premium': round(predicted),
            })
            
            total_premium += predicted
        
        # Apply family discount (already in model, but ensure consistency)
        # The model predicts per-member with family context, so we use the sum
        # Divide by member count since the model already factors in num_members
        adjusted_premium = total_premium / len(members)
        
        # Generate plan tiers
        plan_tiers = [
            {
                'name': 'MediShield Essential',
                'multiplier': 0.75,
                'discount': 32,
                'description': 'Comprehensive plan for essential coverage',
                'features': [
                    'Basic hospitalization cover',
                    'Annual Health Checkup',
                    'Consumables covered',
                    'Day care procedures',
                    'Ambulance charges up to ₹5,000',
                ],
            },
            {
                'name': 'MediShield Premier',
                'multiplier': 1.0,
                'discount': 27,
                'recommended': True,
                'description': 'Comprehensive plan + Additional Riders',
                'features': [
                    'Comprehensive plan + Additional Riders',
                    'Health Checkup',
                    'Consumables covered',
                    '100% hospital bills paid* - No co-payment or room rent capping',
                    'Restoration benefit (100%)',
                    'AYUSH treatment covered',
                    'Pre & Post hospitalization',
                ],
            },
            {
                'name': 'MediShield Premier Plus',
                'multiplier': 1.30,
                'discount': 25,
                'description': f'{sum_insured * 6}L = {sum_insured}L MediShield Premier + {sum_insured * 5}L MediShield Plus',
                'features': [
                    f'{sum_insured * 6}L coverage with super top-up',
                    'Health Checkup',
                    'Consumables covered',
                    '100% hospital bills paid* - No co-payment or room rent capping',
                    'Global coverage',
                    'Air ambulance',
                    'Organ donor expenses',
                    'Maternity cover (add-on)',
                ],
            },
        ]
        
        plans = []
        for tier in plan_tiers:
            premium = round(adjusted_premium * tier['multiplier'])
            original_price = round(premium / (1 - tier['discount'] / 100))
            
            plans.append({
                'name': tier['name'],
                'premium': premium,
                'original_price': original_price,
                'discount': tier['discount'],
                'features': tier['features'],
                'description': tier['description'],
                'recommended': tier.get('recommended', False),
                'gst_amount': round(premium * 0.18),
                'total_with_gst': round(premium * 1.18),
            })
        
        # Save to Database
        try:
            new_inquiry = Inquiry(
                full_name=data.get('fullName', 'Guest'),
                mobile=data.get('mobile', ''),
                email=data.get('email', ''),
                pincode=pincode,
                location=region, # We store the region code or name here
                gender=gender_str,
                members_count=len(members),
                members_json=json.dumps(members),
                conditions_json=json.dumps(conditions),
                is_smoker=bool(is_smoker),
                bmi=float(bmi),
                annual_income=float(data.get('annualIncome', 0)),
                sum_insured=sum_insured,
                tenure=tenure,
                predicted_premium=round(adjusted_premium)
            )
            db.session.add(new_inquiry)
            db.session.commit()
            print(f"📊 Saved inquiry for {new_inquiry.full_name} to database.")
        except Exception as db_err:
            print(f"❌ Database Error: {db_err}")
            db.session.rollback()

        return jsonify({
            'success': True,
            'plans': plans,
            'prediction_details': {
                'member_premiums': member_premiums,
                'total_members': len(members),
                'sum_insured_lacs': sum_insured,
                'tenure_years': tenure,
                'conditions_count': diabetes + hypertension + heart_disease + thyroid + asthma,
                'is_smoker': bool(is_smoker),
                'bmi': float(bmi),
                'region': region,
                'city_tier': city_tier,
                'avg_family_age': round(avg_family_age, 1),
                'risk_factors': get_risk_factors(conditions, is_smoker, bmi, avg_family_age),
            },
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/charts')
def get_charts():
    """Return list of available chart images."""
    charts_dir = 'static/charts'
    if not os.path.exists(charts_dir):
        return jsonify({'charts': []})
    
    charts = []
    for f in os.listdir(charts_dir):
        if f.endswith('.png'):
            charts.append({
                'name': f.replace('.png', '').replace('_', ' ').title(),
                'url': f'/static/charts/{f}',
            })
    
    return jsonify({'charts': charts})


@app.route('/api/model-info')
def model_info():
    """Return model information and metrics."""
    if METRICS is None:
        return jsonify({'loaded': False})
    
    return jsonify({
        'loaded': True,
        'metrics': METRICS,
        'features': FEATURE_COLS,
        'model_type': 'XGBRegressor',
        'training_samples': 50000,
    })


@app.route('/static/charts/<path:filename>')
def serve_chart(filename):
    """Serve chart images."""
    return send_from_directory('static/charts', filename)


# ============================================
# Helper Functions
# ============================================
def get_region_from_pincode(pincode):
    """Determine region code and city tier from pincode."""
    if not pincode or len(pincode) < 2:
        return 0, 2  # Default: Northeast, Tier 2
    
    try:
        first2 = int(pincode[:2])
    except ValueError:
        return 0, 2
    
    # Metro cities (Tier 1)
    metro_pins = {
        11: (0, 1),   # Delhi
        40: (3, 1),   # Mumbai
        56: (2, 1),   # Bangalore
        50: (2, 1),   # Hyderabad
        60: (2, 1),   # Chennai
        70: (0, 1),   # Kolkata
    }
    
    if first2 in metro_pins:
        return metro_pins[first2]
    
    # Region mapping
    if first2 >= 11 and first2 <= 19:
        region = 1  # Northwest
        tier = 2
    elif first2 >= 20 and first2 <= 28:
        region = 1  # Northwest (UP)
        tier = 2 if first2 in [20, 22] else 3
    elif first2 >= 30 and first2 <= 39:
        region = 3  # Southwest (Rajasthan/Gujarat)
        tier = 2
    elif first2 >= 40 and first2 <= 49:
        region = 3  # Southwest (Maharashtra/MP)
        tier = 2
    elif first2 >= 50 and first2 <= 59:
        region = 2  # Southeast (AP/KA)
        tier = 2
    elif first2 >= 60 and first2 <= 69:
        region = 2  # Southeast (TN/KL)
        tier = 2
    elif first2 >= 70 and first2 <= 85:
        region = 0  # Northeast (WB/Bihar/Odisha)
        tier = 3
    else:
        region = 0
        tier = 3
    
    return region, tier


def get_risk_factors(conditions, is_smoker, bmi, avg_age):
    """Generate human-readable risk factors."""
    factors = []
    
    if 'diabetes' in conditions:
        factors.append({'factor': 'Diabetes', 'impact': 'high', 'description': 'Increases premium by ~20%'})
    if 'hypertension' in conditions:
        factors.append({'factor': 'Hypertension', 'impact': 'medium', 'description': 'Increases premium by ~15%'})
    if 'heart' in conditions:
        factors.append({'factor': 'Heart Disease', 'impact': 'very_high', 'description': 'Increases premium by ~35%'})
    if 'thyroid' in conditions:
        factors.append({'factor': 'Thyroid', 'impact': 'low', 'description': 'Increases premium by ~8%'})
    if 'asthma' in conditions:
        factors.append({'factor': 'Asthma', 'impact': 'medium', 'description': 'Increases premium by ~12%'})
    
    if is_smoker:
        factors.append({'factor': 'Smoking', 'impact': 'high', 'description': 'Increases premium by ~25%'})
    
    if bmi and bmi > 30:
        factors.append({'factor': 'High BMI', 'impact': 'medium', 'description': f'BMI {bmi:.1f} increases premium'})
    elif bmi and bmi < 18.5:
        factors.append({'factor': 'Low BMI', 'impact': 'low', 'description': f'BMI {bmi:.1f} slightly increases premium'})
    
    if avg_age > 55:
        factors.append({'factor': 'Senior Members', 'impact': 'high', 'description': 'Higher age group increases premium'})
    
    if not factors:
        factors.append({'factor': 'Low Risk Profile', 'impact': 'positive', 'description': 'No major risk factors detected'})
    
    return factors


# ============================================
# Startup
# ============================================
if __name__ == '__main__':
    print("=" * 60)
    print("MediShield Pro - API Server")
    print("=" * 60)
    
    if load_model():
        print()
        print("Starting server at http://localhost:5000")
        print("   Frontend:  http://localhost:5000/")
        print("   API:       http://localhost:5000/api/predict")
        print("   Health:    http://localhost:5000/api/health")
        print("   Model:     http://localhost:5000/api/model-info")
        print("   Charts:    http://localhost:5000/api/charts")
        print("=" * 60)
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print()
        print("Cannot start server without a trained model.")
        print("   Run: python train_model.py")
        print("=" * 60)
