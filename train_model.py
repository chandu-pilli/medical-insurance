"""
MediShield Pro - ML Model Training Pipeline
Generates synthetic health insurance data and trains an XGBoost model
for accurate insurance cost prediction.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import os
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# 1. Synthetic Data Generation
# ============================================
def generate_insurance_data(n_samples=50000, seed=42):
    """
    Generate realistic synthetic health insurance data.
    Based on real-world insurance pricing patterns in India.
    """
    np.random.seed(seed)
    
    data = []
    
    for _ in range(n_samples):
        # Age: 0-80 years, weighted towards 25-55
        age = int(np.clip(np.random.normal(35, 15), 1, 80))
        
        # Gender: 0 = Male, 1 = Female
        gender = np.random.choice([0, 1], p=[0.52, 0.48])
        
        # BMI: normal distribution centered at 25
        bmi = round(np.clip(np.random.normal(25.5, 5.5), 12, 55), 1)
        
        # Number of children/dependents: 0-5
        num_children = np.random.choice([0, 1, 2, 3, 4, 5], p=[0.20, 0.25, 0.30, 0.15, 0.07, 0.03])
        
        # Smoker: yes/no (15% smokers)
        smoker = np.random.choice([0, 1], p=[0.85, 0.15])
        
        # Region: 0=Northeast, 1=Northwest, 2=Southeast, 3=Southwest
        region = np.random.choice([0, 1, 2, 3])
        
        # Pre-existing conditions (encoded as sum of condition flags)
        diabetes = np.random.choice([0, 1], p=[0.88, 0.12])
        hypertension = np.random.choice([0, 1], p=[0.85, 0.15])
        heart_disease = np.random.choice([0, 1], p=[0.95, 0.05])
        thyroid = np.random.choice([0, 1], p=[0.90, 0.10])
        asthma = np.random.choice([0, 1], p=[0.92, 0.08])
        
        # Medical conditions increase with age
        if age > 50:
            diabetes = np.random.choice([0, 1], p=[0.70, 0.30])
            hypertension = np.random.choice([0, 1], p=[0.65, 0.35])
            heart_disease = np.random.choice([0, 1], p=[0.85, 0.15])
        elif age > 40:
            diabetes = np.random.choice([0, 1], p=[0.80, 0.20])
            hypertension = np.random.choice([0, 1], p=[0.75, 0.25])
        
        # Number of family members insured (1-6)
        num_members = np.random.choice([1, 2, 3, 4, 5, 6], p=[0.15, 0.30, 0.25, 0.15, 0.10, 0.05])
        
        # Sum insured (in lakhs): 3, 5, 7, 10, 15, 20, 25, 50, 75, 100
        sum_insured = np.random.choice(
            [3, 5, 7, 10, 15, 20, 25, 50, 75, 100],
            p=[0.05, 0.15, 0.10, 0.25, 0.15, 0.10, 0.08, 0.07, 0.03, 0.02]
        )
        
        # Policy tenure: 1, 2, 3 years
        tenure = np.random.choice([1, 2, 3], p=[0.60, 0.25, 0.15])
        
        # City tier: 1=Metro, 2=Tier2, 3=Tier3
        city_tier = np.random.choice([1, 2, 3], p=[0.35, 0.40, 0.25])
        
        # Average age of all members
        avg_family_age = age + np.random.normal(0, 10)
        avg_family_age = max(1, min(80, avg_family_age))
        
        # ---- Calculate Premium (realistic formula) ----
        # Base premium per lakh
        if age <= 17:
            base_rate = 600
        elif age <= 25:
            base_rate = 1200
        elif age <= 35:
            base_rate = 2000
        elif age <= 45:
            base_rate = 3200
        elif age <= 55:
            base_rate = 5000
        elif age <= 65:
            base_rate = 8000
        else:
            base_rate = 12000
        
        premium = base_rate * sum_insured
        
        # BMI impact
        if bmi < 18.5:
            premium *= 1.08
        elif bmi > 30:
            premium *= (1 + (bmi - 30) * 0.015)
        elif bmi > 25:
            premium *= (1 + (bmi - 25) * 0.01)
        
        # Smoker surcharge
        if smoker:
            premium *= 1.25
        
        # Conditions multiplier
        conditions_mult = 1.0
        conditions_mult += diabetes * 0.20
        conditions_mult += hypertension * 0.15
        conditions_mult += heart_disease * 0.35
        conditions_mult += thyroid * 0.08
        conditions_mult += asthma * 0.12
        premium *= conditions_mult
        
        # Family members
        premium *= (1 + (num_members - 1) * 0.4)
        
        # Family discount for >2 members
        if num_members >= 4:
            premium *= 0.92
        elif num_members >= 3:
            premium *= 0.95
        
        # Region multiplier
        region_mults = [1.0, 1.03, 1.05, 1.02]
        premium *= region_mults[region]
        
        # City tier
        city_mults = [1.12, 1.04, 1.0]
        premium *= city_mults[city_tier - 1]
        
        # Tenure discount
        tenure_discounts = [1.0, 0.92, 0.85]
        premium *= tenure_discounts[tenure - 1] * tenure
        
        # Gender slight variation
        if gender == 1:  # Female
            premium *= 0.97
        
        # Add some noise (±5%)
        noise = np.random.uniform(0.95, 1.05)
        premium *= noise
        
        premium = max(2000, round(premium))
        
        data.append({
            'age': age,
            'gender': gender,
            'bmi': bmi,
            'num_children': num_children,
            'smoker': smoker,
            'region': region,
            'diabetes': diabetes,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'thyroid': thyroid,
            'asthma': asthma,
            'num_members': num_members,
            'sum_insured': sum_insured,
            'tenure': tenure,
            'city_tier': city_tier,
            'avg_family_age': round(avg_family_age, 1),
            'premium': premium,
        })
    
    df = pd.DataFrame(data)
    return df


# ============================================
# 2. Feature Engineering
# ============================================
def engineer_features(df):
    """Add engineered features to improve model performance."""
    df = df.copy()
    
    # Age bins
    df['age_group'] = pd.cut(df['age'], bins=[0, 17, 25, 35, 45, 55, 65, 100],
                              labels=[0, 1, 2, 3, 4, 5, 6]).astype(int)
    
    # BMI categories
    df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 35, 100],
                                 labels=[0, 1, 2, 3, 4]).astype(int)
    
    # Total conditions count
    df['total_conditions'] = df['diabetes'] + df['hypertension'] + df['heart_disease'] + df['thyroid'] + df['asthma']
    
    # Has any condition
    df['has_condition'] = (df['total_conditions'] > 0).astype(int)
    
    # Age * smoker interaction
    df['age_smoker'] = df['age'] * df['smoker']
    
    # BMI * conditions interaction
    df['bmi_conditions'] = df['bmi'] * df['total_conditions']
    
    # Age * conditions
    df['age_conditions'] = df['age'] * df['total_conditions']
    
    # Sum insured per member
    df['sum_per_member'] = df['sum_insured'] / df['num_members']
    
    # Risk score (composite)
    df['risk_score'] = (df['age'] / 80 * 0.3 +
                        df['bmi'] / 50 * 0.15 +
                        df['smoker'] * 0.2 +
                        df['total_conditions'] / 5 * 0.25 +
                        df['num_members'] / 6 * 0.1)
    
    return df


# ============================================
# 3. Model Training
# ============================================
def train_model(df):
    """Train XGBoost model with hyperparameter tuning."""
    
    # Features and target
    feature_cols = [
        'age', 'gender', 'bmi', 'num_children', 'smoker', 'region',
        'diabetes', 'hypertension', 'heart_disease', 'thyroid', 'asthma',
        'num_members', 'sum_insured', 'tenure', 'city_tier', 'avg_family_age',
        'age_group', 'bmi_category', 'total_conditions', 'has_condition',
        'age_smoker', 'bmi_conditions', 'age_conditions', 'sum_per_member',
        'risk_score'
    ]
    
    X = df[feature_cols]
    y = df['premium']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("=" * 60)
    print("🧠 Training XGBoost Model...")
    print("=" * 60)
    print(f"Training samples: {len(X_train):,}")
    print(f"Testing samples:  {len(X_test):,}")
    print(f"Features:         {len(feature_cols)}")
    print()
    
    # XGBoost model with good hyperparameters
    model = XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    
    # Train with early stopping
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False
    )
    
    # Predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Metrics
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    
    print("📊 Model Performance:")
    print("-" * 40)
    print(f"  Train MAE:   ₹{train_mae:,.0f}")
    print(f"  Test MAE:    ₹{test_mae:,.0f}")
    print(f"  Train RMSE:  ₹{train_rmse:,.0f}")
    print(f"  Test RMSE:   ₹{test_rmse:,.0f}")
    print(f"  Train R²:    {train_r2:.4f}")
    print(f"  Test R²:     {test_r2:.4f}")
    print(f"  CV R² Mean:  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print()
    
    return model, scaler, feature_cols, {
        'train_mae': float(train_mae),
        'test_mae': float(test_mae),
        'train_rmse': float(train_rmse),
        'test_rmse': float(test_rmse),
        'train_r2': float(train_r2),
        'test_r2': float(test_r2),
        'cv_r2_mean': float(cv_scores.mean()),
        'cv_r2_std': float(cv_scores.std()),
    }, X_test, y_test, y_pred_test


# ============================================
# 4. Visualization
# ============================================
def create_visualizations(df, model, feature_cols, scaler, X_test, y_test, y_pred_test):
    """Create analysis charts and save them."""
    
    os.makedirs('static/charts', exist_ok=True)
    
    sns.set_theme(style='whitegrid', palette='husl')
    plt.rcParams['figure.dpi'] = 120
    plt.rcParams['font.family'] = 'sans-serif'
    
    # 1. Actual vs Predicted
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred_test, alpha=0.3, s=10, color='#3b5fff')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Premium (₹)', fontsize=12)
    ax.set_ylabel('Predicted Premium (₹)', fontsize=12)
    ax.set_title('Actual vs Predicted Premium', fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig('static/charts/actual_vs_predicted.png', bbox_inches='tight')
    plt.close()
    
    # 2. Feature Importance
    fig, ax = plt.subplots(figsize=(10, 8))
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({'feature': feature_cols, 'importance': importance})
    feat_imp = feat_imp.sort_values('importance', ascending=True).tail(15)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feat_imp)))
    ax.barh(feat_imp['feature'], feat_imp['importance'], color=colors)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title('Top 15 Feature Importance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('static/charts/feature_importance.png', bbox_inches='tight')
    plt.close()
    
    # 3. Age vs Premium
    fig, ax = plt.subplots(figsize=(8, 6))
    age_premium = df.groupby('age')['premium'].mean()
    ax.plot(age_premium.index, age_premium.values, color='#3b5fff', lw=2)
    ax.fill_between(age_premium.index, age_premium.values, alpha=0.15, color='#3b5fff')
    ax.set_xlabel('Age', fontsize=12)
    ax.set_ylabel('Average Premium (₹)', fontsize=12)
    ax.set_title('Age vs Average Premium', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('static/charts/age_vs_premium.png', bbox_inches='tight')
    plt.close()
    
    # 4. BMI vs Premium
    fig, ax = plt.subplots(figsize=(8, 6))
    bmi_bins = pd.cut(df['bmi'], bins=20)
    bmi_premium = df.groupby(bmi_bins, observed=True)['premium'].mean()
    ax.bar(range(len(bmi_premium)), bmi_premium.values, color='#14b8a6', alpha=0.8)
    ax.set_xlabel('BMI Range', fontsize=12)
    ax.set_ylabel('Average Premium (₹)', fontsize=12)
    ax.set_title('BMI vs Average Premium', fontsize=14, fontweight='bold')
    ax.set_xticks(range(0, len(bmi_premium), 3))
    ax.set_xticklabels([str(x) for x in list(bmi_premium.index)[::3]], rotation=45, ha='right', fontsize=8)
    plt.tight_layout()
    plt.savefig('static/charts/bmi_vs_premium.png', bbox_inches='tight')
    plt.close()
    
    # 5. Smoker vs Non-Smoker
    fig, ax = plt.subplots(figsize=(6, 5))
    smoker_data = df.groupby('smoker')['premium'].mean()
    bars = ax.bar(['Non-Smoker', 'Smoker'], smoker_data.values, color=['#14b8a6', '#ef4444'], width=0.5)
    for bar, val in zip(bars, smoker_data.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500, f'₹{val:,.0f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax.set_ylabel('Average Premium (₹)', fontsize=12)
    ax.set_title('Smoker vs Non-Smoker Premium', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('static/charts/smoker_comparison.png', bbox_inches='tight')
    plt.close()
    
    # 6. Premium Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df['premium'], bins=60, color='#3b5fff', alpha=0.7, edgecolor='white')
    ax.axvline(df['premium'].median(), color='#ef4444', linestyle='--', lw=2, label=f"Median: ₹{df['premium'].median():,.0f}")
    ax.set_xlabel('Premium (₹)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Premium Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('static/charts/premium_distribution.png', bbox_inches='tight')
    plt.close()
    
    # 7. Conditions Impact
    fig, ax = plt.subplots(figsize=(8, 5))
    conditions = ['diabetes', 'hypertension', 'heart_disease', 'thyroid', 'asthma']
    cond_labels = ['Diabetes', 'Hypertension', 'Heart Disease', 'Thyroid', 'Asthma']
    cond_impact = []
    for c in conditions:
        with_cond = df[df[c] == 1]['premium'].mean()
        without_cond = df[df[c] == 0]['premium'].mean()
        impact_pct = ((with_cond - without_cond) / without_cond) * 100
        cond_impact.append(impact_pct)
    
    colors = ['#f97316', '#ef4444', '#dc2626', '#8b5cf6', '#06b6d4']
    bars = ax.bar(cond_labels, cond_impact, color=colors, width=0.5)
    for bar, val in zip(bars, cond_impact):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'+{val:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax.set_ylabel('Premium Increase (%)', fontsize=12)
    ax.set_title('Impact of Pre-existing Conditions on Premium', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('static/charts/conditions_impact.png', bbox_inches='tight')
    plt.close()
    
    # 8. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    corr_cols = ['age', 'bmi', 'smoker', 'diabetes', 'hypertension', 'heart_disease',
                 'num_members', 'sum_insured', 'tenure', 'city_tier', 'premium']
    corr_matrix = df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                square=True, ax=ax, linewidths=0.5)
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('static/charts/correlation_heatmap.png', bbox_inches='tight')
    plt.close()
    
    print("📈 Charts saved to static/charts/")


# ============================================
# 5. Save Model & Artifacts
# ============================================
def save_artifacts(model, scaler, feature_cols, metrics):
    """Save trained model and preprocessing artifacts."""
    os.makedirs('models', exist_ok=True)
    
    # Save model
    joblib.dump(model, 'models/xgb_insurance_model.pkl')
    print("✅ Model saved: models/xgb_insurance_model.pkl")
    
    # Save scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    print("✅ Scaler saved: models/scaler.pkl")
    
    # Save feature columns
    with open('models/feature_cols.json', 'w') as f:
        json.dump(feature_cols, f)
    print("✅ Feature columns saved: models/feature_cols.json")
    
    # Save metrics
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("✅ Metrics saved: models/metrics.json")


# ============================================
# Main Pipeline
# ============================================
if __name__ == '__main__':
    print("=" * 60)
    print("🏥 MediShield Pro - ML Training Pipeline")
    print("=" * 60)
    print()
    
    # Step 1: Generate data
    print("📊 Step 1: Generating synthetic insurance data...")
    df = generate_insurance_data(n_samples=50000)
    print(f"   Generated {len(df):,} samples")
    print(f"   Features: {list(df.columns)}")
    print(f"   Premium range: ₹{df['premium'].min():,} — ₹{df['premium'].max():,}")
    print(f"   Premium mean:  ₹{df['premium'].mean():,.0f}")
    print(f"   Premium median: ₹{df['premium'].median():,.0f}")
    print()
    
    # Save raw data
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/insurance_data.csv', index=False)
    print(f"   💾 Data saved to data/insurance_data.csv")
    print()
    
    # Step 2: Feature engineering
    print("🔧 Step 2: Engineering features...")
    df_engineered = engineer_features(df)
    print(f"   Total features: {len(df_engineered.columns)}")
    print()
    
    # Step 3: Train model
    print("🧠 Step 3: Training model...")
    model, scaler, feature_cols, metrics, X_test, y_test, y_pred_test = train_model(df_engineered)
    
    # Step 4: Save artifacts
    print("💾 Step 4: Saving model artifacts...")
    save_artifacts(model, scaler, feature_cols, metrics)
    print()
    
    # Step 5: Visualizations
    print("📈 Step 5: Creating visualizations...")
    create_visualizations(df_engineered, model, feature_cols, scaler, X_test, y_test, y_pred_test)
    print()
    
    print("=" * 60)
    print("✅ Training pipeline complete!")
    print(f"   Model R² Score: {metrics['test_r2']:.4f}")
    print(f"   Model MAE: ₹{metrics['test_mae']:,.0f}")
    print("   Run 'python app.py' to start the API server")
    print("=" * 60)
