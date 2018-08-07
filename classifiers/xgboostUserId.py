"""
cb, 07.08.2018

 - user identification (with non walking data) using xgboost

"""

from abraxas4.abraxasFrame import AbraxasFrame
xgbAbra = AbraxasFrame(numIrSensors=10, numFrSensors=2, windowWidth=100, windowShift=100, numCoeffs=5, numFreqs=1,
                     enaStatFeats=True, wavelet='haar', waveletLvl1=False, featNormMethod='stand', trainFraction=0.66,
                     classSortTT=True, randomSortTT=False, lineThresholdAfterNorm=10, enaRawFeats=False, corrPeaks=2)

xgbAbra.loadTeTrDump(dumpName='xgbAbra.pkl')

from xgboost import XGBClassifier
clf = XGBClassifier(booster="gbtree", eta=0.3, max_depth=3)
xgbAbra.trainClassifier(classifier=clf, supervised=True)
xgbAbra.testClassifier()

