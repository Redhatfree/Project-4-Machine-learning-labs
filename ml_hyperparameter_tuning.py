"""
Hyperparameter Tuning, Pipeline, and ROC/AUC Evaluation
Author: [Your Name]
Description: Implementing an end-to-end classification pipeline using GridSearchCV,
             evaluating performance with ROC-AUC, and preparing code for production.
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    # -------------------------------------------------------------------------
    # 1. DATA PREPARATION
    # -------------------------------------------------------------------------
    print("=== Loading and Preparing Data ===")
    wine = load_wine()
    
    # Converting to a binary classification problem for ROC/AUC: 
    # Class 1 vs Others (Class 0 and Class 2)
    X = wine.data
    y = (wine.target == 1).astype(int) 
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Feature Scaling (Crucial for consistent pipeline design)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # -------------------------------------------------------------------------
    # 2. HYPERPARAMETER TUNING VIA GRID SEARCH
    # -------------------------------------------------------------------------
    print("\n=== Executing GridSearchCV ===")
    
    # Define the model
    rf = RandomForestClassifier(random_state=42)
    
    # Define the hyperparameter grid to explore
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5]
    }
    
    # n_jobs=-1 utilizes all available CPU cores for parallel processing
    grid_search = GridSearchCV(
        estimator=rf, 
        param_grid=param_grid, 
        cv=5, 
        scoring='roc_auc', 
        n_jobs=-1
    )
    
    # Fit the grid search to find the best parameters
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"Best Hyperparameters Found: {grid_search.best_params_}")
    print(f"Best Cross-Validation ROC-AUC Score: {grid_search.best_score_:.4f}")
    
    # -------------------------------------------------------------------------
    # 3. PRODUCTION RESOURCE ANALYSIS (CV RESULTS TO DATAFRAME)
    # -------------------------------------------------------------------------
    print("\n=== Processing CV Results Table ===")
    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    
    # Extracting key metrics to evaluate the Bias-Variance tradeoff
    metrics_summary = cv_results_df[['param_max_depth', 'param_n_estimators', 'mean_test_score', 'std_test_score']]
    print(metrics_summary.sort_values(by='mean_test_score', ascending=False).head())
    
    # -------------------------------------------------------------------------
    # 4. MODEL EVALUATION (ROC & AUC)
    # -------------------------------------------------------------------------
    print("\n=== Evaluating Test Set Performance ===")
    best_model = grid_search.best_estimator_
    
    # Predict probabilities for the positive class (class 1)
    y_probs = best_model.predict_proba(X_test_scaled)[:, 1]
    y_preds = best_model.predict(X_test_scaled)
    
    # Calculate operational metrics
    auc_score = roc_auc_score(y_test, y_probs)
    print(f"Test Set AUC Score: {auc_score:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_preds))
    
    # -------------------------------------------------------------------------
    # 5. VISUALIZATION (Plotting the ROC Curve)
    # -------------------------------------------------------------------------
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    # Save the plot automatically so it can be added to your GitHub README
    plt.savefig('roc_curve_output.png', dpi=300)
    print("\nVisual asset 'roc_curve_output.png' successfully generated.")

if __name__ == "__main__":
    main()
