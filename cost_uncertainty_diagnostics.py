import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# 1. SIMULATE ROAD CONSTRUCTION DATASET
# ==========================================
np.random.seed(42)
n_projects = 500

baseline_budget = np.random.uniform(50, 500, size=n_projects)
project_length_km = np.random.uniform(2, 45, size=n_projects)
ground_risk_index = np.random.uniform(1, 10, size=n_projects)

actual_cost = (
    baseline_budget 
    + (project_length_km * 2.5) 
    + (ground_risk_index ** 1.8) * 3 
    + np.random.normal(0, 15, size=n_projects)
)

df = pd.DataFrame({
    "BaselineBudget_MNOK": baseline_budget,
    "ProjectLength_KM": project_length_km,
    "GroundRiskIndex": ground_risk_index,
    "ActualCost_MNOK": actual_cost
})

# Missing data injection (~6%)
df.loc[np.random.rand(n_projects) < 0.06, "GroundRiskIndex"] = np.nan

X = df.drop("ActualCost_MNOK", axis=1)
y = df["ActualCost_MNOK"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# 2. TRAIN PIPELINE
# ==========================================
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)

# ==========================================
# 3. FEATURE IMPORTANCE DIAGNOSTICS
# ==========================================
rf_model = pipeline.named_steps["model"]
importances = rf_model.feature_importances_
feature_names = X.columns

print("--- Feature Importance Analysis ---")
for name, importance in zip(feature_names, importances):
    print(f"{name}: {importance * 100:.2f}%")

# ==========================================
# 4. SAVE MODEL & RUN INFERENCE
# ==========================================
# Save pipeline to disk
model_filename = "cost_risk_model.joblib"
joblib.dump(pipeline, model_filename)
print(f"\nModel successfully saved to '{model_filename}'")

# Load pipeline back from disk
loaded_pipeline = joblib.load(model_filename)

# New project query: Budget = 250 MNOK, Length = 15 km, Ground Risk = 8.5
new_project = pd.DataFrame({
    "BaselineBudget_MNOK": [250.0],
    "ProjectLength_KM": [15.0],
    "GroundRiskIndex": [8.5]
})

predicted_cost = loaded_pipeline.predict(new_project)[0]
print(f"\nEstimated Final Cost for New Project: {predicted_cost:.2f} MNOK")