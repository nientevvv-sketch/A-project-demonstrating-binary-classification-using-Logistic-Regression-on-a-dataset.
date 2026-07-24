# A-project-demonstrating-binary-classification-using-Logistic-Regression-on-a-dataset.
Logistic Regression is a fundamental linear classification algorithm that models the probability of binary outcomes. It uses the sigmoid function to map linear combinations of features to probabilities between 0 and 1, making it ideal for binary classification tasks.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

X = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
y = np.array([0,0,0,0,1,1,1,1,1,1])



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)



model =LogisticRegression()
model.fit(X_train, y_train) 

y_pred = model.predict(X_test)
print("Предсказания:", y_pred)
print("Истинные значения:", y_test)



accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

