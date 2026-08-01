import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. SIMULATE THE DATA
np.random.seed(42)
n_samples = 500

# Features on massive different scales
age = np.random.randint(18, 80, size=n_samples)
total_charges = np.random.uniform(100, 8000, size=n_samples)
churn = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])

df = pd.DataFrame({"Age": age, "TotalCharges": total_charges, "Churn": churn})

# Inject missing values into TotalCharges
df.loc[np.random.rand(n_samples) < 0.08, "TotalCharges"] = np.nan

# Split into Features and Target
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. BUILD THE PIPELINE
# TODO: Define a list of steps that does three things in sequence:
#       First: Imputes missing values using the "median" strategy.
#       Second: Standardizes the features using StandardScaler.
#       Third: Initializes a KNeighborsClassifier with n_neighbors=5.
steps = [
    # ("imputer", ...),
    # ("scaler", ...),
    # ("knn", ...)
]

pipeline = Pipeline(steps)

# 3. TRAIN AND EVALUATE
# TODO: Fit the complete pipeline to the training data
# pipeline.fit(...)

# TODO: Generate predictions on the test set
# y_pred = pipeline.predict(...)

# PRINT DIAGNOSTICS
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))