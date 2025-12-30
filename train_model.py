import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load dataset
df = pd.read_csv("diet_recommendations_dataset.csv")

# Features & Target
X = df.drop(columns=["Diet_Recommendation", "Patient_ID"])
y = df["Diet_Recommendation"]

# Categorical & Numerical columns
categorical_cols = [
    "Gender", "Disease_Type", "Severity",
    "Physical_Activity_Level", "Dietary_Restrictions",
    "Allergies", "Preferred_Cuisine"
]

numerical_cols = [
    "Age", "Weight_kg", "Height_cm", "BMI",
    "Daily_Caloric_Intake", "Cholesterol_mg/dL",
    "Blood_Pressure_mmHg", "Glucose_mg/dL",
    "Weekly_Exercise_Hours", "Adherence_to_Diet_Plan",
    "Dietary_Nutrient_Imbalance_Score"
]

# Encoders
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Scaler
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Enhanced Model with better parameters
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("=" * 50)
print("🎯 MODEL TRAINING COMPLETE")
print("=" * 50)
print(f"\n✅ Accuracy: {accuracy:.2%}")
print(f"✅ Features: {len(X.columns)}")
print(f"✅ Training Samples: {len(X_train)}")
print(f"✅ Test Samples: {len(X_test)}")
print("\n" + "=" * 50)

# Save everything
joblib.dump(model, "diet_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump({
    'accuracy': accuracy,
    'feature_importance': dict(zip(X.columns, model.feature_importances_)),
    'categorical_cols': categorical_cols,
    'numerical_cols': numerical_cols
}, "model_metadata.pkl")

print("\n💾 All files saved successfully!")
print("   - diet_model.pkl")
print("   - label_encoders.pkl")
print("   - scaler.pkl")
print("   - model_metadata.pkl")
print("\n🚀 Ready to launch Streamlit app!")