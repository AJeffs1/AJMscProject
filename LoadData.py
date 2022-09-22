import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import CameraCalibration
import math

#..\Dataset\M4p-c0-T.josn

def xydist(xyPair1, xyPair2):
    return math.sqrt(((xyPair1[0] - xyPair2[0]) ** 2) + ((xyPair1[1] - xyPair2[1]) ** 2))

def loadcsvdata():
    print('start load csv data')
    df0 = pd.read_csv("YoloV4Out/M4p-c0-M.csv")
    df1 = pd.read_csv("YoloV4Out/M4p-c1-M.csv")
    df2 = pd.read_csv("YoloV4Out/M4p-c2-M.csv")
    df3 = pd.read_csv("YoloV4Out/M4p-c3-M.csv")
    #print(list(df0))  ## ['frame_id', 'object_id', 'class', 'x', 'y', 'center', 'xmin', 'ymin', 'xmax', 'ymax']
    #df = df["framesp1"] # just person 1
    #df = df[0]

    OutC0People = pd.DataFrame(columns=['frame_id', 'object_id', 'xmin', 'ymin', 'xmax', 'ymax', 'RWRefX', 'RWRefY']) # Out for each camera
    OutC1People = pd.DataFrame(columns=['frame_id', 'object_id', 'xmin', 'ymin', 'xmax', 'ymax', 'RWRefX', 'RWRefY'])
    OutC3People = pd.DataFrame(columns=['frame_id', 'object_id', 'xmin', 'ymin', 'xmax', 'ymax', 'RWRefX', 'RWRefY'])

    lastKnownPeople = []

    # standard form of xy,
    for Frame in range(3,901):
        temppeopleco = [[],[],[],[]]
        rows0 = df0.loc[df0['frame_id'] == Frame]
        rows1 = df1.loc[df1['frame_id'] == Frame]
        #rows2 = df0.loc[df0['frame_id'] == Frame]
        rows3 = df3.loc[df3['frame_id'] == Frame]
        #Row0People = [] # df of frame id, person id, xy min and max, RWR calculated
        Row0People = pd.DataFrame(columns=['frame_id', 'object_id', 'xmin', 'ymin', 'xmax', 'ymax', 'RWRefX', 'RWRefY'])
        Row1People = pd.DataFrame(columns=['frame_id', 'object_id', 'xmin', 'ymin', 'xmax', 'ymax', 'RWRefX', 'RWRefY'])
        #Row2People = pd.DataFrame(columns=['frame_id', 'object_id', 'xmin', 'ymin', 'xmax', 'ymax', 'RWRefX', 'RWRefY'])
        Row3People = pd.DataFrame(columns=['frame_id', 'object_id', 'xmin', 'ymin', 'xmax', 'ymax', 'RWRefX', 'RWRefY'])
        for Row in range(rows0.shape[0]):
            CurrentPerson = rows0.iloc[Row]# pull out nessesary info for check.
            Point = (CurrentPerson['ymax'] , (CurrentPerson['xmax'] + CurrentPerson['xmin']) / 2)# get B Middle Cordinates
            PR = CameraCalibration.Camera0Generalise(Point[0], Point[1]) # AKA RWR
            Row0People = Row0People.append({'frame_id': CurrentPerson['frame_id'], 'object_id': CurrentPerson['object_id'], 'xmin': CurrentPerson['xmin'], 'ymin': CurrentPerson['ymin'], 'xmax' : CurrentPerson['xmax'], 'ymax' : CurrentPerson['ymax'], 'RWRefX' : PR[0], 'RWRefY' : PR[1]}, ignore_index= True)

        for Row in range(rows1.shape[0]):
            CurrentPerson = rows1.iloc[Row]# pull out nessesary info for check.
            Point = (CurrentPerson['ymax'] , (CurrentPerson['xmax'] + CurrentPerson['xmin']) / 2)# get B Middle Cordinates
            PR = CameraCalibration.Camera1Generalise(Point[0], Point[1]) # AKA RWR
            Row1People = Row1People.append({'frame_id': CurrentPerson['frame_id'], 'object_id': CurrentPerson['object_id'],'xmin': CurrentPerson['xmin'], 'ymin': CurrentPerson['ymin'], 'xmax': CurrentPerson['xmax'],'ymax': CurrentPerson['ymax'], 'RWRefX': PR[0], 'RWRefY': PR[1]}, ignore_index=True)

        for Row in range(rows3.shape[0]):
            CurrentPerson = rows3.iloc[Row]# pull out nessesary info for check.
            Point = (CurrentPerson['ymax'] , (CurrentPerson['xmax'] + CurrentPerson['xmin']) / 2)# get B Middle Cordinates
            PR = CameraCalibration.Camera3Generalise(Point[0], Point[1]) # AKA RWR
            Row3People = Row3People.append({'frame_id': CurrentPerson['frame_id'], 'object_id': CurrentPerson['object_id'],'xmin': CurrentPerson['xmin'], 'ymin': CurrentPerson['ymin'], 'xmax': CurrentPerson['xmax'],'ymax': CurrentPerson['ymax'], 'RWRefX': PR[0], 'RWRefY': PR[1]}, ignore_index=True)

        # C0      xmin	ymin	xmax	ymax
        FirstFrameLocations = [[1, 0.19716503, 0.490029417], [2, 0.452178553, 0.57412324], [3, -0.50947166, 0.435294477]] # 1 is P1
        #FirstFrameLocations = [[1, 0.452178553, 0.57412324], [2, 0.19716503, 0.490029417],[3, -0.50947166, 0.435294477]]  # 1 is P1
        P4Start = 141 # this manual value is just for GT testing. as gives point of references as it starts with the same people
        P4Location = [4, 1.6975105, -1.2070811]

        if Frame == 3:
            lastKnownPeople = FirstFrameLocations

        if Frame == P4Start:
            lastKnownPeople.append(P4Location)
            temppeopleco[3].append([1.6975105, -1.2070811])
        ## Section of finding unified object IDs --------------------------------------------------------------
        ## Start C0
        pointinrow = -1  # declaration
        CorrectID = -1 # fail safe
        taken = []
        C0Changes = []
        for PLoc in range(Row0People.shape[0]):
            FoundP = Row0People.iloc[PLoc] # current person being checked
            tempDist = 10000 # temp distance to beat
            tempLoc = []
            for KnownP in lastKnownPeople: # for all known people
                dist = xydist([FoundP['RWRefX'], FoundP['RWRefX']], [KnownP[1], KnownP[2]])
                #print(f"Place in list is {PLoc}   ,Know person is :{KnownP[0]}")
                if (dist < tempDist) and (KnownP[0] not in taken): # if they match and are not taken
                    tempDist = dist
                    pointinrow = PLoc
                    tempLoc = [FoundP['RWRefX'], FoundP['RWRefX']]
                    CorrectID = KnownP[0]
            taken.append(CorrectID)
            #print(taken)
            temppeopleco[CorrectID - 1].append(tempLoc)
            C0Changes.append([pointinrow, CorrectID])# pairs should be C0s index, object id

        ## Next C1

        pointinrow = -1  # declaration
        CorrectID = -1  # fail safe
        taken = []
        C1Changes = []
        for PLoc in range(Row1People.shape[0]):
            FoundP = Row1People.iloc[PLoc]
            tempLoc = []
            tempDist = 10000
            for KnownP in lastKnownPeople:
                dist = xydist([FoundP['RWRefX'], FoundP['RWRefX']], [KnownP[1], KnownP[2]])
                if (dist < tempDist) and (KnownP[0] not in taken):
                    tempDist = dist
                    pointinrow = PLoc
                    tempLoc = [FoundP['RWRefX'], FoundP['RWRefX']]
                    CorrectID = KnownP[0]
            taken.append(CorrectID)
            temppeopleco[CorrectID - 1].append(tempLoc)
            C1Changes.append([pointinrow, CorrectID])  # pairs should be C0s index, object id

        ## Next C3

        pointinrow = -1  # declaration
        CorrectID = -1  # fail safe
        taken = []
        tempLoc = []
        C3Changes = []
        for PLoc in range(Row3People.shape[0]):
            FoundP = Row3People.iloc[PLoc]
            tempDist = 10000
            for KnownP in lastKnownPeople:
                dist = xydist([FoundP['RWRefX'], FoundP['RWRefX']], [KnownP[1], KnownP[2]])
                if (dist < tempDist) and (KnownP[0] not in taken):
                    tempDist = dist
                    pointinrow = PLoc
                    tempLoc = [FoundP['RWRefX'], FoundP['RWRefX']]
                    CorrectID = KnownP[0]
            taken.append(CorrectID)
            temppeopleco[CorrectID - 1].append(tempLoc)
            C3Changes.append([pointinrow, CorrectID])  # pairs should be C0s index, object id

        ## Section End -------------------------------------------------------------------------------
        ## Section make adjustments to the object ids -------------------------------------------------------------------------------
        for tempPair in C0Changes:
            Row0People.at[tempPair[0], 'object_id'] = tempPair[1]
        for tempPair in C1Changes:
            Row1People.at[tempPair[0], 'object_id'] = tempPair[1]
        for tempPair in C3Changes:
            Row3People.at[tempPair[0], 'object_id'] = tempPair[1]

        OutC0People = OutC0People.append(Row0People,ignore_index=True)
        OutC1People = OutC1People.append(Row1People,ignore_index=True)
        OutC3People = OutC3People.append(Row3People,ignore_index=True)

        nextturnxy = []
        #print(f"len of last know people  {len(lastKnownPeople)}")
        #print(lastKnownPeople)
        #print(f"Len of Temp people {len(temppeopleco)}")
        #print(temppeopleco)

        for p in range(len(temppeopleco)):
            tempxtotal = 0
            tempytotal = 0
            tempnum = 0
            if temppeopleco[p]:
                for c in range(len(temppeopleco[p])):
                    if temppeopleco[p][c]:
                        #print(f"pre add Tempx total {tempxtotal}")
                        tempxtotal += temppeopleco[p][c][0]
                        #print(f"Tempx total {tempxtotal}")
                        tempytotal += temppeopleco[p][c][1]
                        #print(f"tempytotal total {tempytotal}")
                        tempnum +=1
                        #print(f"tempnum  {tempnum}")
                nextturnxy.append([tempxtotal / tempnum , tempytotal / tempnum])

        #print(nextturnxy)
        #print(lastKnownPeople)
        lastKnownPeople = [] # saves last known people avg for all found people
        for p in range(len(nextturnxy)):
            lastKnownPeople.append([p + 1, nextturnxy[p][0], nextturnxy[p][1]])

        ## Section End -------------------------------------------------------------------------------


    #print('FIN')
    #OutC0People.to_pickle('OutC0PeopleV2.pkl') # saves in df form
    #OutC1People.to_pickle('OutC1PeopleV2.pkl')
    #OutC3People.to_pickle('OutC3PeopleV2.pkl')
    #OutC0People.to_csv("OutC0PeopleV2.csv", sep=',')
