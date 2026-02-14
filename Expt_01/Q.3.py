# Q.3 To understand and implement data splitting techniques for machine learning models.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Create sample dataset
data = {
    'Height': [150, 160, 170, np.nan, 180],
    'Weight': [50, 60, 70, 65, 80],
    'Age': [20, 25, 30, 28, 35],
    'Fitness': ['Low', 'Medium', 'High', 'Medium', 'High']
}

df = pd.DataFrame(data)

# 2. Handling missing values
df['Height'] = df['Height'].fillna(df['Height'].mean())

# 3. Encoding categorical data
df['Fitness'] = df['Fitness'].map({'Low': 0, 'Medium': 1, 'High': 2})

# 4. Feature Scaling
scaler = StandardScaler()
df[['Height', 'Weight', 'Age']] = scaler.fit_transform(
    df[['Height', 'Weight', 'Age']]
)

# 5. Splitting Features and Target
X = df[['Height', 'Weight', 'Age']]
y = df['Fitness']

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

# Display Results
print("--- Processed Dataframe ---")
print(df)

print("\n--- Split Sizes ---")
print("Training data size:", X_train.shape)
print("Testing data size:", X_test.shape)
