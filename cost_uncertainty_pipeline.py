import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 1. SIMULATE ROAD CONSTRUCTION COST DATA
# ==========================================
np.random.seed(42)
n_projects = 500

# Features: Baseline Budget (MNOK), Project Length (km), Ground Risk Index (1-10)
baseline_budget = np.random.uniform(50, 500, size=n_projects)
project_length_km = np.random.uniform(2, 45, size=n_projects)
ground_risk_index = np.random.uniform(1, 10, size=n_projects)

# Calculate Actual Final Cost with non-linear risk interaction + random variance
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

# Inject missing values (~6%) into GroundRiskIndex to simulate incomplete reports
df.loc[np.random.rand(n_projects) < 0.06, "GroundRiskIndex"] = np.nan

# Separate features (X) and target variable (y)
X = df.drop("ActualCost_MNOK", axis=1)
y = df["ActualCost_MNOK"]

# Train/Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# 2. BUILD REGRESSION ML PIPELINE
# ==========================================
# Impute missing risk indexes -> Scale features -> Random Forest Regressor
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])

# ==========================================
# 3. TRAIN & EVALUATE MODEL
# ==========================================
# Fit pipeline on training data
pipeline.fit(X_train, y_train)

# Predict actual costs on test set
y_pred = pipeline.predict(X_test)

# Calculate regression evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("--- Norwegian Road Project Cost Prediction ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} MNOK")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} MNOK")
print(f"R-squared Score (R²): {r2:.4f}")