"""
cb, 07.08.2018

 - user identification (with non walking data) using decision tree alg

"""

from abraxas4.abraxasFrame import AbraxasFrame

dtAbra = AbraxasFrame(numIrSensors=10, numFrSensors=2, windowWidth=100, windowShift=25, numFreqs=1, numCoeffs=0,
                      enaStatFeats=True, featNormMethod='stand', trainFraction=2/3, waveletLvl1=False,
                      randomSortTT=False, classSortTT=True, corrPeaks=0, enaRawFeats=False)

dtAbra.loadTeTrDump(dumpName='dtAbra.pkl')

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion="entropy", max_depth=20, min_samples_leaf=1, min_samples_split=3,
                             max_features=None, max_leaf_nodes=None)
dtAbra.trainClassifier(classifier=clf)
dtAbra.testClassifier()

