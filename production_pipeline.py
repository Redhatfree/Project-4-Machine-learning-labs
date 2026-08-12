import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# ==========================================
# STEP 1: SIMULATE AND CLEAN RAW DATA
# ==========================================
# Set random seed for reproducibility
np.random.seed(42)
n_samples = 1000

# Generate 5 numerical features
X_num = np.random.randn(n_samples, 5)

# Generate an imbalanced target array (approx. 90% Class 0, 10% Class 1)
y_imbalanced = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])

# Combine into a single Pandas DataFrame
columns = [f"feature_{i}" for i in range(5)]
df = pd.DataFrame(X_num, columns=columns)
df["target"] = y_imbalanced

# Introduce random missing values (NaNs) into feature_0 to simulate real-world data
nan_mask = np.random.rand(n_samples) < 0.05  # ~5% missingness
df.loc[nan_mask, "feature_0"] = np.nan

# Audit missing values
print("--- Missing Values Per Column ---")
print(df.isna().sum())
print("-" * 35)

# Separate features (X) and target (y)
X = df.drop("target", axis=1)
y = df["target"]

# Stratified split to ensure the 90/10 class balance is identical in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# STEP 2: ASSEMBLE THE UNIFIED PIPELINE
# ==========================================
# We chain the imputer and the regularized model into a single pipeline object.
# class_weight="balanced" forces the model to heavily penalize misclassifying the minority class.
pipeline_steps = [
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", LogisticRegression(class_weight="balanced", solver="liblinear", random_state=42))
]

model_pipeline = Pipeline(pipeline_steps)

# ==========================================
# STEP 3: AUTOMATE HYPERPARAMETER TUNING
# ==========================================
# Define the hyperparameter search space. 
# Note the prefix "classifier__" to tell the pipeline which step these settings belong to.
param_distributions = {
    "classifier__C": np.logspace(-4, 4, 20),
    "classifier__penalty": ["l1", "l2"]
}

# Wrap the pipeline inside RandomizedSearchCV to isolate the cross-validation folds safely
random_search = RandomizedSearchCV(
    estimator=model_pipeline,
    param_distributions=param_distributions,
    n_iter=10,
    cv=5,
    scoring="roc_auc",  # Optimizing for AUC instead of deceptive raw accuracy
    n_jobs=-1,          # Utilize all available CPU cores
    random_state=42
)

# Run the automated search engine using only training data (No data leakage!)
random_search.fit(X_train, y_train)

# ==========================================
# STEP 4: EXECUTE AND EVALUATE
# ==========================================
print("\n--- Hyperparameter Optimization ---")
print(f"Best Parameters Found: {random_search.best_params_}")
print(f"Best Cross-Validation ROC-AUC: {random_search.best_score_:.4f}")
print("-" * 35)

# Evaluate on the completely untainted test set
final_model = random_search.best_estimator_
y_pred = final_model.predict(X_test)
y_pred_proba = final_model.predict_proba(X_test)[:, 1]

print("\n--- Final Performance Metrics ---")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print(f"Test Set ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
print("-" * 35)