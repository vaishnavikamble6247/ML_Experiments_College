import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Load dataset
data = pd.read_csv('house_price.csv')

# 2. Features and Target
X = data[['Area']]
y = data['Price']

# 3. Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predict
y_pred = model.predict(X_test)

# 6. Evaluate using RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", rmse)

# 7. Visualization
plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, color='green', label='Actual Price')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')

plt.title('House Price vs Area')
plt.xlabel('Area (sq ft)')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
