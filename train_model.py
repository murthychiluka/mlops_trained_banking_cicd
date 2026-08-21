# ----------------------------
# train_model_real_data.py
# ----------------------------

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
import random

# ----------------------------
# 1️⃣ Reproducibility
# ----------------------------
random.seed(42)
np.random.seed(42)

# ----------------------------
# 2️⃣ Load Real Dataset
# ----------------------------
# Make sure your CSV has columns: 
# age, income, loan_amount, loan_term, credit_score, gender, married, approved
df = pd.read_csv("loan_data.csv")

# ----------------------------
# 3️⃣ Separate Features and Labels
# ----------------------------
X = df[['age', 'income', 'loan_amount', 'loan_term', 'credit_score', 'gender', 'married']]
y = df['approved']

# ----------------------------
# 4️⃣ Encode Categorical Columns
# ----------------------------
le_gender = LabelEncoder()
le_married = LabelEncoder()
le_approved = LabelEncoder()

X['gender'] = le_gender.fit_transform(X['gender'])
X['married'] = le_married.fit_transform(X['married'])
y_encoded = le_approved.fit_transform(y)

# ----------------------------
# 5️⃣ Split Data for Training & Testing
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# ----------------------------
# 6️⃣ Train Random Forest Model
# ----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

# ----------------------------
# 7️⃣ Evaluate Model
# ----------------------------
accuracy = model.score(X_test, y_test)
print(f"✅ Model trained successfully with accuracy: {accuracy:.2f}")

# ----------------------------
# 8️⃣ Store Feature Metadata
# ----------------------------
feature_names = ['age', 'income', 'loan_amount', 'loan_term', 'credit_score', 'gender', 'married']
model.expected_features = feature_names

# ----------------------------
# 9️⃣ Save Model & Encoders
# ----------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/model_full.pkl")
joblib.dump(le_gender, "models/encoder_gender.pkl")
joblib.dump(le_married, "models/encoder_married.pkl")
joblib.dump(le_approved, "models/encoder_approved.pkl")

print("💾 Model and encoders saved in ./models/")
print(f"📦 Model expects features: {model.expected_features}")
