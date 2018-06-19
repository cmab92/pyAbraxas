from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from pandas import read_csv
from datetime import datetime
# load data
def parse(x):
	return datetime.strptime(x, '%Y %m %d %H')
from abraxasOne.loadAndClean import loadAndClean
allData = loadAndClean("ankita.txt", 10, 2, tSample = 0.0165, dirPath = "")
allData = allData[0]
dataset = []
dataset.append(allData[0][100:2100,0])
for i in range(len(allData)):
    dataset.append(allData[i][100:2100,1])

dataset = {'s0': dataset[1], 's1': dataset[2], 's2': dataset[3], 's3': dataset[4], 's4': dataset[5], 's5': dataset[6], 's6': dataset[7], 's7': dataset[8], 's8': dataset[9], 's9': dataset[10]}
# mark all NA values with 0
dataset = pd.DataFrame(dataset)
#dataset['irData'].fillna(0, inplace=True)
print(dataset.head(5))
# save to file# manually specify column names
dataset.drop([1], axis=0)
print(dataset.head(5))
# mark all NA values with 0

#dataset.to_csv('irData.csv')
dataset.to_csv("" + "irData2.csv", sep=',', index=False, header=False)

