import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_and_export(pipeline, X_test, y_test, feature_names, output_path):
    # Make predictions
    predictions = pipeline.predict(X_test)

    # Calculate regression metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("--- Model Performance ---")
    print(f"MAE:  {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R^2:  {r2:.4f}")

    # Extract feature importances from Random Forest regressor
    regressor = pipeline.named_steps["regressor"]
    importances = regressor.feature_importances_

    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values(by="Importance", ascending=False)

    print("\n--- Feature Importances ---")
    print(importance_df.to_string(index=False))

    # Serialize fitted pipeline artifact
    joblib.dump(pipeline, output_path)
    print(f"\nPipeline successfully saved to {output_path}")


if __name__ == "__main__":
    # Example execution (replace with your actual pipeline and split data)
    # evaluate_and_export(fitted_pipeline, X_test, y_test, feature_cols, 'models/regression_pipeline.joblib')
    pass