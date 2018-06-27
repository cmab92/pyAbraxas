"""
cb, 27.06.2018

 - receive data via COM port and format data (specifically re-arrange bno055 data)

 Detailed:
 The input data contains data read from arm-analog-ports (infrared sensors and force sensors) and bno055 data (linear
 acceleration, angular velocity and quaternion data). Given is a row of unprocessed values, e. g.:

 ir_0, ir_1, ..., ir_m, force_0, ..., force_n, bno_0, bno_1, bno_2, bno_3, index
 where: 0 <= ir_i, force_j <= 1023 and -2^14 <= bno_k <= 2^14

 The bno data has a lower data rate, to allow higher sampling rates. 'index' indicates the type of data:
 index = 0 := quaternion data
 index = 1 := linear acceleration data
 index = 2 := angular velocity data

 This function receives all data and formats it, s. t. the output data is given as:
 ir_0, ir_1, ..., ir_m, force_0, ..., force_n, q_w, q_x, q_y, q_z, la_x, la_y, la_z, av_y, av_y, av_z
 where: 0 <= ir_i, force_j <= 1 and -1 <= bno_k <= 1

 Incomplete frames are interpolated. A time axis is provided.

 Received data is returned and written to a .txt-file.

 """
def receiveData(port=None, baudRate=57600):

