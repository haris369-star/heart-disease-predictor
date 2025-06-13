# ------------------------- app.py -------------------------

import pandas as pd
import numpy as np
import os
import warnings
import sys
from flask import Flask, request
from flask_cors import CORS

from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, BaggingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Suppress warnings, especially from LightGBM
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['LIGHTGBM_DISABLE_STDERR_REDIRECTION'] = '1'
sys.stderr = open(os.devnull, 'w')

# ---------------------- Model Setup

# Define dataset path (ensure the CSV is in the 'data' folder)
dataset_path = os.path.join(os.getcwd(), 'data', 'heart_statlog_cleveland_hungary_final (1).csv')
df = pd.read_csv(dataset_path)
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Basic cleaning
df['chest_pain_type'] = df['chest_pain_type'] - 1
df['st_slope'] = df['st_slope'] - 1
df['oldpeak'] = df['oldpeak'].clip(lower=0, upper=6)
df = df.drop_duplicates()

# Feature Engineering
df['age_maxhr_interaction'] = df['age'] * df['max_heart_rate']
df['oldpeak_cp_interaction'] = df['oldpeak'] * (df['chest_pain_type'] + 1)
df['st_slope_exang_interaction'] = (df['st_slope'] + 1) * (df['exercise_angina'] + 1)

# Separate features and target
target = 'target'
X = df.drop(columns=[target])
y = df[target]

# Scaling using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Feature Selection using SelectKBest (chi²) with k=15
selector = SelectKBest(score_func=chi2, k=15)
X_selected = selector.fit_transform(X_scaled, y)
selected_features = pd.DataFrame(X_scaled, columns=X.columns).columns[selector.get_support()].tolist()

# Train-Test Split (70/30) with stratification on the target
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, stratify=y, test_size=0.3, random_state=42
)

# Data Balancing using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Model Building:
# LightGBM model
lgbm_best = LGBMClassifier(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42)
lgbm_best.fit(X_train_resampled, y_train_resampled)

# XGBoost model
xgb_best = XGBClassifier(
    n_estimators=300, learning_rate=0.01, max_depth=5, use_label_encoder=False,
    eval_metric='logloss', random_state=42
)
xgb_best.fit(X_train_resampled, y_train_resampled)

# Soft Voting Ensemble of LightGBM and XGBoost
voting_clf = VotingClassifier(estimators=[
    ('lgbm', lgbm_best),
    ('xgb', xgb_best)
], voting='soft')
voting_clf.fit(X_train_resampled, y_train_resampled)

# Bagging wrapper to reduce variance further
bagging_voting = BaggingClassifier(estimator=voting_clf, n_estimators=10, random_state=42)
bagging_voting.fit(X_train_resampled, y_train_resampled)

print("✅ Model Training Completed Successfully!")

# ---------------------- Flask App Setup

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def home():
    return "✅ Heart Disease Prediction API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # Convert input to DataFrame
        input_df = pd.DataFrame([data])
        # Feature Engineering for input data
        input_df['age_maxhr_interaction'] = input_df['age'] * input_df['max_heart_rate']
        input_df['oldpeak_cp_interaction'] = input_df['oldpeak'] * (input_df['chest_pain_type'] + 1)
        input_df['st_slope_exang_interaction'] = (input_df['st_slope'] + 1) * (input_df['exercise_angina'] + 1)
        # Reorder columns to match the model's expected input features
        input_df = input_df[selected_features]
        # Apply scaling
        input_scaled = scaler.transform(input_df)
        # Apply feature selection
        input_selected = selector.transform(input_scaled)
        # Model prediction
        prediction = bagging_voting.predict(input_selected)[0]
        prob_heart_attack = bagging_voting.predict_proba(input_selected)[:, 1][0]
        # Format output
        if prediction == 0:
            return "No Risk of Heart Attack"
        else:
            risk = "LOW Risk" if prob_heart_attack < 0.5 else "MODERATE Risk" if prob_heart_attack < 0.8 else "HIGH Risk"
            return f"Risk of Heart Attack\n{round(prob_heart_attack * 100, 2)}%\n{risk}"
    except Exception as e:
        return f"Error: {str(e)}"

# Run the Flask app; use environment variable PORT if available (for Render) or fallback to 8000
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))
