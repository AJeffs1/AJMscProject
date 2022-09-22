import cv2 as cv
import glob
import json
import pandas as pd
import numpy as np
import h5py


def loaddata():
    print('start load data')
    f = open('GroundTruth/M4p-c0-T.json')
    data = json.load(f)
    df = pd.DataFrame(data["annotations"])
    #print(list(df))
    df = df["framesp1"] # just person 1
    df = df[0]

    #print(list(df))
    print(type(df))
    ser = pd.Series(df)
    print(ser.head(10))
    print(ser[0])
    print(ser[1])


def buildfloorplane():
    print('start build floor plane')
    C0 = CalcC0()
    C1 = CalcC1()
    C2 = CalcC2()
    C3 = CalcC3()

    # takes all 4 coordinate sets
    print(C0)
    #print(C1)
    #print(C2)
    #print(C3)

    print(C0['y'][0])
    test = ZPlaneCalc(C0['x'][0],C0['y'][0],0)
    print(test)


    # Z coordinates of rugs ?

    # open cv camera calibration 0 and 3 , 1 (2 bad)



def CalcC0():
    f = open('GroundTruth/M4p-c0-T.json')
    data = json.load(f)
    df = pd.DataFrame(data["annotations"])
    #print(list(df))
    dftl = df["framesTL"]  # just one class TL
    dfbl = df["framesBL"]  # just person 1

    #(list(dftl[4]))
    dftl = dftl[4]
    dftl = dftl['0']
    #print(dftl['text'])
    dftl = dftl['bounding_box']
    tlh = dftl['h']
    tlw = dftl['w']
    tlx = dftl['x']
    tly = dftl['y']

    # then other image
    #print('second BB')
    dfbl = dfbl[5]
    dfbl = dfbl['0']
    #print(dfbl['text'])
    dfbl = dfbl['bounding_box']
    blh = dfbl['h']
    blw = dfbl['w']
    blx = dfbl['x']
    bly = dfbl['y']
    # order is TL ,TR ,BL , BR
    datat = {'x': [tlx, blx + blw, blx, tlx + tlw], 'y': [tly + tlh, bly, bly + blh, tly]}
    C0 = pd.DataFrame.from_dict(datat)
    return C0

def CalcC1():
    f = open('GroundTruth/M4p-c1-T.json')
    data = json.load(f)
    df = pd.DataFrame(data["annotations"])
    #print(list(df))
    dftl = df["framesTL"]  # just one class TL
    dfbl = df["framesBL"]  # just person 1

    #print(list(dftl[6]))
    dftl = dftl[6]
    dftl = dftl['0']
    #print(dftl['text'])
    dftl = dftl['bounding_box']
    tlh = dftl['h']
    tlw = dftl['w']
    tlx = dftl['x']
    tly = dftl['y']

    # then other image
    #print('second BB')
    dfbl = dfbl[5]
    dfbl = dfbl['0']
    #print(dfbl['text'])
    dfbl = dfbl['bounding_box']
    blh = dfbl['h']
    blw = dfbl['w']
    blx = dfbl['x']
    bly = dfbl['y']
    # order is TL ,TR ,BL , BR
    dataT = {'x': [tlx + tlw, blx, blx + blw, tlx], 'y': [tly + tlh, bly, bly + blh, tly]}
    C1 = pd.DataFrame.from_dict(dataT)
    #print(C1)
    return C1

def CalcC2():
    f = open('GroundTruth/M4p-c2-T.json')
    data = json.load(f)
    df = pd.DataFrame(data["annotations"])
    #print(list(df))
    dftl = df["framesTL"]  # just one class TL
    dfbl = df["framesBL"]  # just person 1

    #(list(dftl[6]))
    dftl = dftl[6]
    dftl = dftl['0']
    #print(dftl['text'])
    dftl = dftl['bounding_box']
    tlh = dftl['h']
    tlw = dftl['w']
    tlx = dftl['x']
    tly = dftl['y']

    # then other image
    #print('second BB')
    dfbl = dfbl[5]
    dfbl = dfbl['0']
    #print(dfbl['text'])
    dfbl = dfbl['bounding_box']
    blh = dfbl['h']
    blw = dfbl['w']
    blx = dfbl['x']
    bly = dfbl['y']
    # order is TL ,TR ,BL , BR
    datat = {'x': [100000, 100000, blx + blw, tlx], 'y': [100000, 100000, bly, tly]}
    C2 = pd.DataFrame.from_dict(datat)
    return C2

def CalcC3():
    f = open('GroundTruth/M4p-c3-T.json')
    data = json.load(f)
    df = pd.DataFrame(data["annotations"])
    #print(list(df))
    dftl = df["framesTL"]  # just one class TL
    dfbl = df["framesBL"]  # just person 1
    #print(list(dftl))
    #(list(dftl[10]))
    dftl = dftl[10]
    dftl = dftl['0']
    #print(dftl['text'])
    dftl = dftl['bounding_box']
    tlh = dftl['h']
    tlw = dftl['w']
    tlx = dftl['x']
    tly = dftl['y']

    # then other image
    #print('second BB')
    dfbl = dfbl[9]
    dfbl = dfbl['0']
    #print(dfbl['text'])
    dfbl = dfbl['bounding_box']
    blh = dfbl['h']
    blw = dfbl['w']
    blx = dfbl['x']
    bly = dfbl['y']
    # order is TL ,TR ,BL , BR
    datat = {'x': [tlx, blx + blw, blx, tlx + tlw], 'y': [tly, bly + blh, bly, tly + tlh]}
    C3= pd.DataFrame.from_dict(datat)
    return C3

def ZPlaneCalc(X,Y,C): # must be re written, taken from my old DISS
    if (C == 0):
        p1 = (0.176138,     0.647589, - 63.412272) #Camera 0
        p2 = (-0.180912,    0.622446, - 0.125533)
        p3 = (-0.000002,    0.001756, 0.102316)
    elif (C == 1):
        p1 = (0.177291,	    0.004724,	31.224545) # Camera 1
        p2 = (0.169895,	    0.661935,	-79.781865)
        p3 = (-0.000028,    0.001888,	0.054634)
    elif (C == 2):
        p1 = (-0.118791,	0.077787,	64.819189) # Camera 2
        p2 = (0.133127,	    0.069884,	15.832922)
        p3 = (-0.000001,	0.002045,	-0.057759,)
    else:
        p1 = (-0.142865,	0.553150,	-17.395045) #Camera 3
        p2 = (-0.125726,	0.039770,	75.937144)
        p3 = (-0.000011,	0.001780,	0.015675)
    x = X
    y = Y
    v1 = (p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])
    v2 = (p1[0] - p3[0], p1[1] - p3[1], p1[2] - p3[2])
    n = np.cross(v1, v2)  # 2
    k = ((n[0] * p1[0]) + (n[1] * p1[1]) + (n[2] * p1[2]))  # 3
    z = 1 / n[2] * ((n[0] * p1[0]) + (n[1] * p1[1]) + (n[2] * p1[2]) - (n[0] * x) - (n[1] * y))  # 4
    return z



def oldloaddata():
    print('start load data')
    f = open('GroundTruth/M4p-c0-T.json')
    data = json.load(f)
    df = pd.DataFrame(data["annotations"])
    #print(list(df))
    df = df["framesp1"] # just person 1
    df = df[0]

    #print(list(df))
    print(type(df))
    ser = pd.Series(df)
    print(ser.head(10))
    print(ser[0])
    print(ser[1])


