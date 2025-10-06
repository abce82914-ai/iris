# model.py
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# Load Iris dataset
iris = load_iris()
X = iris.data  # features: sepal/petal length/width
y = iris.target  # labels: 0 = setosa, 1 = versicolor, 2 = virginica

# Split into train/test (not strictly necessary for saving, but good practice)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save the trained model to a file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained on Iris dataset and saved as model.pkl")
