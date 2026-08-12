import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# 1. Load Data
# 
df = pd.read_csv('dataset.csv')

# 2. Handling Missing Values 
# 
imputer = SimpleImputer(strategy='mean')
df['age'] = imputer.fit_transform(df[['age']])

# 3. Outlier Detection using IQR 
# 
Q1 = df['income'].quantile(0.25)
Q3 = df['income'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 
df_clean = df[(df['income'] >= lower_bound) & (df['income'] <= upper_bound)]

# 4. Feature Scaling 
# ب
scaler = StandardScaler()
numerical_cols = ['age', 'income', 'spending_score']
df_clean[numerical_cols] = scaler.fit_transform(df_clean[numerical_cols])

# 5. Save Cleaned Data
df_clean.to_csv('cleaned_dataset.csv', index=False)
print("Data preprocessing completed successfully and saved!")