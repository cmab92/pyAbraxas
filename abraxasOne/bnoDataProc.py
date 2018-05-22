########################################################################################################################
## cb, 22.05.18
##
## general info:
##
##
## function inputs:
## linAccData: linAccData[i][j,k] -> i-th coordinate (x,y,z), j-th data point (k=0 -> time (in s), k=1 -> sensor-value (in m/s^2))
## angVecData: angVecData[i][j,k] -> i-th coordinate (x,y,z), j-th data point (k=0 -> time (in s), k=1 -> sensor-value (deg/s))
## quatData: quatData[i][j,k] -> i-th coordinate (w,x,y,z), j-th data point (k=0 -> time (in s), k=1 -> sensor-value (normalized |value|=<1 ))
##
## functions output:
## data: data[i,j]:= i-th data point of j-th sensor
########################################################################################################################
def bnoDataProc(linAccData, angVecData, quatData):

    return 0