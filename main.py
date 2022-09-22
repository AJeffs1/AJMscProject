
# AJ MSC Python code 17636888
import LoadData
import ZCalculation
import CameraCalibration
import os
import Evaluation
import pandas as pd

print('begin')

#LoadData.loadcsvdata()  # Load in dataset and run system on it
Evaluation.evaluationProcess()  # run evaluation on the dataset

C0T = CameraCalibration.Camera0Generalise(432.15, 309.71) # AKA RWR
C1T = CameraCalibration.Camera1Generalise(457.8, 569.74) # AKA RWR
C3T = CameraCalibration.Camera3Generalise(801.0, 370.03) # AKA RWR
print("test from another cordinate")
print(C0T)
print(C1T)
print(C3T)

C0T = CameraCalibration.Camera0Generalise(202.07, 334.02) # from same coordinate
C1T = CameraCalibration.Camera1Generalise(465.51, 901.14) # AKA RWR
C3T = CameraCalibration.Camera3Generalise(726.97, 287.5) # AKA RWR

print("test from same coordinated at calibration point")
print(C0T)
print(C1T)
print(C3T)


print('Finished')

#Times of Running

# object detection and tracking Yolo system on google colab
# 112.711 C0
# 117.062 C1
# 114.537 C2
# 118.194 C3

# Pose estimation on google colab
#C0	CPU times: user 11min 19s, sys: 29.1 s, total: 11min 48s Wall time: 6min 52s
#C1	CPU times: user 11min 19s, sys: 29 s, total: 11min 48s Wall time: 6min 47s
#C2	CPU times: user 11min 19s, sys: 26.9 s, total: 11min 46s Wall time: 6min 43s
#C3	CPU times: user 11min 20s, sys: 26.6 s, total: 11min 46s Wall time: 6min 42s

# The solutions tracking correction
#--- 137.4814488887787 seconds ---
#--- 135.02752113342285 seconds ---

# the evaluation program
#--- 70.34997534751892 seconds ---
#--- 69.74871444702148 seconds ---