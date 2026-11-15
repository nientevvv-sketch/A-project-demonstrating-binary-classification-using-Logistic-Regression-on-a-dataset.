# A project demonstrating binary classification using Logistic Regression.
# Logistic Regression estimates the probability of one of two possible classes.
# It applies the sigmoid function to map a weighted sum of input features to a probability.

# Split the data into training and testing utilities.
from sklearn.model_selection import train_test_split
# Import the Logistic Regression classifier.
from sklearn.linear_model import LogisticRegression
# Import metrics for evaluating classification results.
from sklearn.metrics import accuracy_score, confusion_matrix
# Import NumPy to create and reshape the example data.
import numpy as np

# Create a simple dataset with one numeric feature and binary labels (0 or 1).
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# Reserve 30% of the examples for testing; use a fixed seed for reproducibility.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create the classifier and train it using the training examples.
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict the binary class for each feature value in the test set.
y_pred = model.predict(X_test)
print("Предсказания:", y_pred)
print("Истинные значения:", y_test)

# Calculate the fraction of test predictions that match the true labels.
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Show counts of true and predicted labels for each class.
# Rows represent true labels; columns represent predicted labels.
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
