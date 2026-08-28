import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from typing import Dict, Tuple

class SyntheticProjectDataGenerator:
    """Generates synthetic historical project data with non-linear cost escalation."""
    @staticmethod
    def generate(n_samples: int = 200, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
        np.random.seed(random_state)
        size_m2 = np.random.uniform(500, 10000, n_samples)
        team_exp_years = np.random.uniform(1, 15, n_samples)
        complexity_score = np.random.randint(1, 6, n_samples)
        duration_months = size_m2 / 500 + np.random.uniform(2, 12, n_samples)

        # Base cost formula ($ Millions)
        base_cost = (size_m2 * 0.0012) + (complexity_score * 0.45) - (team_exp_years * 0.05)
        
        # Heteroscedastic noise (cost overrun risk increases with project size & complexity)
        noise_scale = 0.10 + (size_m2 / 10000) * (complexity_score / 3.0)
        cost_overrun_noise = np.random.gamma(shape=2.0, scale=noise_scale, size=n_samples)
        
        actual_cost = base_cost + cost_overrun_noise

        df = pd.DataFrame({
            'size_m2': size_m2,
            'team_exp_years': team_exp_years,
            'complexity_score': complexity_score,
            'duration_months': duration_months
        })
        return df, pd.Series(actual_cost, name='actual_cost_mUSD')

class ProbabilisticCostEstimator:
    """
    Trains multiple quantile regressors to predict P10, P50, and P90 cost estimates
    for project risk management.
    """
    def __init__(self, quantiles: list = [0.10, 0.50, 0.90]):
        self.quantiles = quantiles
        self.models: Dict[float, Pipeline] = {}

    def _build_pipeline(self, quantile: float) -> Pipeline:
        return Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', GradientBoostingRegressor(
                loss='quantile',
                alpha=quantile,
                n_estimators=100,
                max_depth=3,
                random_state=42
            ))
        ])

    def fit(self, X: pd.DataFrame, y: pd.Series):
        for q in self.quantiles:
            pipeline = self._build_pipeline(q)
            pipeline.fit(X, y)
            self.models[q] = pipeline
        return self

    def predict_p_values(self, X: pd.DataFrame) -> pd.DataFrame:
        predictions = {}
        for q, model in self.models.items():
            predictions[f'P{int(q*100)}'] = model.predict(X)
        
        results_df = pd.DataFrame(predictions, index=X.index)
        
        # Calculate Contingency Buffer required (P90 - P50)
        results_df['Contingency_P90_Buffer'] = results_df['P90'] - results_df['P50']
        return results_df

if __name__ == "__main__":
    # 1. Generate Training Data
    X_train, y_train = SyntheticProjectDataGenerator.generate(n_samples=500)

    # 2. Train Quantile Estimators (P10, P50, P90)
    estimator = ProbabilisticCostEstimator(quantiles=[0.10, 0.50, 0.90])
    estimator.fit(X_train, y_train)

    # 3. Predict for a New Enterprise Project Scope
    # Feature inputs: Size: 4,500 m2 | Team Exp: 3 yrs | Ground/Site Complexity: 4/5 | Duration: 14 mos
    new_project = pd.DataFrame([{
        'size_m2': 4500,
        'team_exp_years': 3,
        'complexity_score': 4,
        'duration_months': 14
    }])

    cost_estimates = estimator.predict_p_values(new_project)

    print("==================================================")
    print("      PROJECT PROBABILISTIC COST ESTIMATE         ")
    print("==================================================")
    print(f"P10 Estimate (Optimistic) : ${cost_estimates['P10'].values[0]:.3f} M")
    print(f"P50 Estimate (Median)     : ${cost_estimates['P50'].values[0]:.3f} M")
    print(f"P90 Estimate (Conservative): ${cost_estimates['P90'].values[0]:.3f} M")
    print("--------------------------------------------------")
    print(f"Recommended Contingency Buffer (P90 - P50): ${cost_estimates['Contingency_P90_Buffer'].values[0]:.3f} M")
    print("==================================================")