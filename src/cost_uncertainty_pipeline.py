import subprocess
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline


def create_quantile_pipeline(alpha: float) -> Pipeline:
    """Builds a Scikit-Learn Pipeline with Gradient Boosting Quantile Loss."""
    numeric_features = ["project_length", "ground_risk_score", "tunneling_ratio"]
    categorical_features = ["project_type"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = GradientBoostingRegressor(
        loss="quantile",
        alpha=alpha,
        n_estimators=100,
        random_state=42
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])


def push_to_git(commit_message: str = "feat: add quantile regression pipeline artifact") -> None:
    """Automates staging, committing, and pushing changes to GitHub."""
    try:
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print("\nSuccessfully committed and pushed code to GitHub!")
    except subprocess.CalledProcessError as e:
        print(f"\nGit automation failed: {e}")


def main():
    # Synthetic dataset matching project schema
    data = pd.DataFrame({
        "project_length": [12.5, 8.0, 22.1, 5.4, 18.3],
        "ground_risk_score": [0.8, 0.3, 0.9, 0.2, 0.6],
        "tunneling_ratio": [0.4, 0.0, 0.6, 0.1, 0.3],
        "project_type": ["road", "bridge", "tunnel", "road", "tunnel"],
        "final_cost": [150.0, 85.0, 310.0, 60.0, 220.0]
    })

    X = data.drop(columns=["final_cost"])
    y = data["final_cost"]

    # Train Quantile Pipeline for P50 target estimate
    p50_pipeline = create_quantile_pipeline(alpha=0.50)
    p50_pipeline.fit(X, y)

    # Save model artifact locally
    model_path = "models/quantile_p50_pipeline.joblib"
    joblib.dump(p50_pipeline, model_path)
    print(f"Model successfully saved to {model_path}")

    # Push to remote repository
    push_to_git(commit_message="feat: implement quantile pipeline and export p50 model")


if __name__ == "__main__":
    main()