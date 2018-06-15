import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score# load data
from abraxasOne.helperFunctions import powerSetOfArray
from mpl_toolkits.mplot3d import Axes3D

dataset1 = np.loadtxt('irData1.csv', delimiter=",")
dataset2 = np.loadtxt('irData2.csv', delimiter=",")

label1 = np.zeros(len(dataset1))
label2 = np.ones(len(dataset2))

X = np.concatenate([dataset1, dataset2])


Y = np.concatenate([label1, label2])

seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

ps = powerSetOfArray(np.array([2, 3, 4, 5, 6, 7, 8, 9]))
#for j in range(len(ps)):
ps = np.array([[5, 7, 0], [5, 7, 1], [5, 7, 2], [5, 7, 3], [5, 7, 4], [5, 7, 6], [5, 7, 8], [5, 7, 9]])
for j in range(9):
    if len(ps[j]) == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(int(len(X_train)/19)):
            if y_train[i]==1:
                ax.scatter(X_train[i][ps[j][0]], X_train[i][ps[j][1]], X_train[i][ps[j][2]], c='r', marker='o')
        for i in range(int(len(X_train)/19)):
            if y_train[i]==0:
                ax.scatter(X_train[i][ps[j][0]], X_train[i][ps[j][1]], X_train[i][ps[j][2]], c='b', marker='^')
        plt.title(str(ps[j][0]) + str(ps[j][1]) + str(ps[j][2]))
    plt.show()

model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))