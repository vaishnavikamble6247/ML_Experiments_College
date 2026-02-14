import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# 1. Load dataset
data = pd.read_csv('product_purchase.csv')

# 2. Features and Target
X = data[['Age', 'Salary']]
y = data['Bought'].map({'No': 0, 'Yes': 1})

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. Train Decision Tree Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 5. Predict
y_pred = model.predict(X_test)

# 6. Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 7. Simple Visualization
plt.scatter(data['Age'], data['Salary'], c=y, cmap='coolwarm')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('Product Purchase Classification')
plt.colorbar(label='Bought (0=No, 1=Yes)')
plt.show()
