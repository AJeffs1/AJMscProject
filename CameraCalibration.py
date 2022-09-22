#!/usr/bin/env python

# ref docs
import cv2
import numpy as np

# these functions are used to calculate the real world generalisesd xy of the people
def Camera0Generalise(X, Y):
    objpoints0 = []  # Vector for 3D
    imgpoints0 = []  # Vector for 2D
    img0 = cv2.imread("images/C0IMG.png")  # read in image to check for sized
    gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    # 2d coordinates
    preimgpoints0 = np.array([[202.07, 334.02], [516.06, 227.44], [307.17, 437.96], [651.9, 288.23]], dtype=np.float32)
    preobjpoints0 = np.array([[0, 0, 0], [0, 205, 0], [130, 0, 0], [130, 205, 0]], dtype=np.float32)
    objpoints0.append(preobjpoints0)  # Convert arrays correct format
    imgpoints0.append(preimgpoints0)
    # mtx is camera matrix, Does calibration process
    ret0, mtx0, dist0, rvecs0, tvecs0 = cv2.calibrateCamera(objpoints0, imgpoints0, gray0.shape[::-1], None, None, )
    # Ground plane homography
    H0 = np.array([[0.176138, 0.647589, -63.412272], [-0.180912, 0.622446, -0.125533], [-0.000002, 0.001756, 0.102316]],dtype=np.float32)
    num0, Rs0, Ts0, Ns0 = cv2.decomposeHomographyMat(H0, mtx0)  # calculated Info from homography,rotation and distance
    #X,Y calculation
    formattedxy = np.array([X, Y], dtype=np.float32)  # test coordinates
    xy_undistorted0 = cv2.undistortPoints(formattedxy, mtx0, dist0)
    xy_undistorted0 = xy_undistorted0[0,0]
    return xy_undistorted0

def Camera1Generalise(X, Y):
    objpoints1 = []  # Vector for 3D
    imgpoints1 = []  # Vector for 2D , these are the locations of the rug in set coordinates
    img1 = cv2.imread("images/C1IMG.png")  # read in image to check for sized
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # 2d coordinates
    preimgpoints1 = np.array([[465.51, 901.14], [191.81, 510.77], [794.92, 631.93], [452.2, 399.32]], dtype=np.float32)
    preobjpoints1 = np.array([[0, 0, 0], [0, 205, 0], [130, 0, 0], [130, 205, 0]], dtype=np.float32)
    objpoints1.append(preobjpoints1)  # Convert arrays correct format
    imgpoints1.append(preimgpoints1)
    # mtx is camera matrix, Does calibration process
    ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints1, imgpoints1, gray1.shape[::-1], None, None, )
    # Ground plane homography
    H1 = np.array([[0.177291, 0.004724, 31.224545], [0.169895, 0.661935, -79.781865], [-0.000028, 0.001888, 0.054634]],dtype=np.float32)
    num1, Rs1, Ts1, Ns1 = cv2.decomposeHomographyMat(H1, mtx1)  # calculated Info from homography,rotation and distance
    #X,Y calculation
    formattedxy = np.array([X, Y], dtype=np.float32)  # test coordinates
    xy_undistorted1 = cv2.undistortPoints(formattedxy, mtx1, dist1)
    xy_undistorted1 = xy_undistorted1[0, 0]
    return xy_undistorted1

def Camera2Generalise(X, Y):
    objpoints0 = []  # Vector for 3D
    imgpoints0 = []  # Vector for 2D , these are the locations of the rug in set coordinates
    img0 = cv2.imread("images/C2IMG.png")  # read in image to check for sized
    gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    # 2d coordinates
    preimgpoints0 = np.array([[465.51, 901.14], [191.81, 510.77], [794.92, 631.93], [452.2, 399.32]], dtype=np.float32) # not updated
    preobjpoints0 = np.array([[0, 0, 0], [0, 205, 0], [130, 0, 0], [130, 205, 0]], dtype=np.float32)
    objpoints0.append(preobjpoints0)  # Convert arrays correct format
    imgpoints0.append(preimgpoints0)
    # mtx is camera matrix, Does calibration process
    ret0, mtx0, dist0, rvecs0, tvecs0 = cv2.calibrateCamera(objpoints0, imgpoints0, gray0.shape[::-1], None, None, )
    # Ground plane homography
    H0 = np.array([[-0.118791, 0.077787, 64.819189], [0.133127, 0.069884, 15.832922], [-0.000001, 0.002045, -0.057759]],dtype=np.float32)
    num0, Rs0, Ts0, Ns0 = cv2.decomposeHomographyMat(H0, mtx0)  # calculated Info from homography,rotation and distance
    #X,Y calculation
    formattedxy = np.array([X, Y], dtype=np.float32)  # test coordinates
    xy_undistorted0 = cv2.undistortPoints(formattedxy, mtx0, dist0)
    xy_undistorted0 = xy_undistorted0[0, 0]
    return xy_undistorted0

def Camera3Generalise(X, Y):
    objpoints0 = []  # Vector for 3D
    imgpoints0 = []  # Vector for 2D , these are the locations of the rug in set coordinates
    img0 = cv2.imread("images/C3IMG.png")  # read in image to check for sized
    gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    # 2d coordinates
    preimgpoints0 = np.array([[726.97, 287.5], [1060.19, 403.92], [560.82, 339.31], [923.9, 511.14]], dtype=np.float32)
    preobjpoints0 = np.array([[0, 0, 0], [0, 205, 0], [130, 0, 0], [130, 205, 0]], dtype=np.float32)
    objpoints0.append(preobjpoints0)  # Convert arrays correct format
    imgpoints0.append(preimgpoints0)
    # mtx is camera matrix, Does calibration process
    ret0, mtx0, dist0, rvecs0, tvecs0 = cv2.calibrateCamera(objpoints0, imgpoints0, gray0.shape[::-1], None, None, )
    # Ground plane homography
    H0 = np.array([[-0.142865, 0.553150, -17.395045], [-0.125726, 0.039770, 75.937144], [-0.000011, 0.001780, 0.015675]],dtype=np.float32)
    num0, Rs0, Ts0, Ns0 = cv2.decomposeHomographyMat(H0, mtx0)  # calculated Info from homography,rotation and distance
    #X,Y calculation
    formattedxy = np.array([X, Y], dtype=np.float32)  # test coordinates
    xy_undistorted0 = cv2.undistortPoints(formattedxy, mtx0, dist0)
    xy_undistorted0 = xy_undistorted0[0, 0]
    return xy_undistorted0


