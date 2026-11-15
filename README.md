# Logistic Regression Binary Classification Demo

This project demonstrates a simple binary classification task using Logistic Regression from scikit-learn.

Logistic Regression is a linear model used for classification problems where the target variable has two possible outcomes. It applies the sigmoid function to convert a linear combination of input features into a probability between 0 and 1, which is then used to classify the sample.

## Project goal

The goal of this example is to:
- prepare a small dataset,
- split it into training and test sets,
- train a Logistic Regression model,
- make predictions,
- evaluate the model using accuracy and a confusion matrix.

## Requirements

Before running the project, make sure you have installed:

- Python 3.x
- NumPy
- scikit-learn

Install dependencies with:

```bash
pip install numpy scikit-learn
```

## Example code

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Predictions:", y_pred)
print("Actual values:", y_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
```

## How it works

1. The data is created as a simple 1D feature set.
2. The dataset is split into training and testing subsets.
3. A Logistic Regression model is trained on the training data.
4. The model predicts the test labels.
5. The predictions are evaluated with:
   - accuracy score,
   - confusion matrix.

## Output

The script prints:
- predicted labels,
- actual labels,
- model accuracy,
- confusion matrix for evaluating classification performance.

This is a minimal educational example of logistic regression for binary classification.
