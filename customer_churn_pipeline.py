import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 1. SIMULATE THE DATASET
# ==========================================
np.random.seed(42)
n_samples = 500

# Generate features and target variable
age = np.random.randint(18, 80, size=n_samples)
total_charges = np.random.uniform(100, 8000, size=n_samples)
churn = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])

df = pd.DataFrame({
    "Age": age,
    "TotalCharges": total_charges,
    "Churn": churn
})

# Introduce ~8% missing values into TotalCharges
df.loc[np.random.rand(n_samples) < 0.08, "TotalCharges"] = np.nan

# Separate features (X) and target (y)
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train/Test Split (80% train, 20% test) with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# 2. BUILD THE MACHINE LEARNING PIPELINE
# ==========================================
# Sequential steps: Impute missing values -> Scale features -> Train KNN
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])

# ==========================================
# 3. TRAIN & EVALUATE THE MODEL
# ==========================================
# Fit the pipeline on training data
pipeline.fit(X_train, y_train)

# Make predictions on test data
y_pred = pipeline.predict(X_test)

# Display evaluation diagnostics
print("--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))