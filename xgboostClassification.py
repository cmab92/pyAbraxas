import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score# load data


dataset1 = np.loadtxt('irData1.csv', delimiter=",")
dataset2 = np.loadtxt('irData2.csv', delimiter=",")

label1 = np.zeros(len(dataset1))
label2 = np.ones(len(dataset2))

X = np.concatenate([dataset1, dataset2])


Y = np.concatenate([label1, label2])

seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

for i in range(len(X_test)):
    if y_train[i]==1:
        plt.scatter(X_train[i], mark='o')
for i in range(len(X_test)):
    if y_train[i]==0:
        plt.scatter(X_train[i], mark='^')
# fit model no training data
plt.show()

model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))