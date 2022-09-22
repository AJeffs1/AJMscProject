
import math
from shapely.geometry import Polygon
import pandas as pd
import json

def calculate_iou(boxA, boxB):
    polyA = Polygon(boxA) #foramt box = [[511, 41], [577, 41], [577, 76], [511, 76]]
    polyB = Polygon(boxB)
    iou = (polyA.intersection(polyB).area) / (polyA.union(polyB).area)
    return iou

def same_BB(B1, B2):
    dist = math.sqrt(((B1[0] - B2[0]) ** 2) + ((B1[1] - B2[1]) ** 2))#B1 = [300,200]
    if (dist < 40):
        outcome = True
    else:
        outcome = False

    return outcome

def xydist(xyPair1, xyPair2):
    return math.sqrt(((xyPair1[0] - xyPair2[0]) ** 2) + ((xyPair1[1] - xyPair2[1]) ** 2))

def evaluationProcess():
    #Load up all GT for C0
    #Load up all values from system
    #Load up data output from yolov4

    emptyRow = pd.Series([{'bounding_box': {'h': -1, 'w': -1, 'x': -1, 'y': -1}, 'instance_id': {'value': -1}, 'keyframe': False}])

    SystemC0df = pd.read_pickle('OutputData/OutC0PeopleV1.pkl')
    SystemC1df = pd.read_pickle('OutputData/OutC1PeopleV1.pkl')
    #SystemC2df = pd.read_pickle('OutputData/OutC2PeopleV1.pkl')
    SystemC3df = pd.read_pickle('OutputData/OutC3PeopleV1.pkl')

    YoloC0df = pd.read_csv("YoloV4Out/M4p-c0-M.csv")
    YoloC1df = pd.read_csv("YoloV4Out/M4p-c1-M.csv")
    #YoloC2df = pd.read_csv("YoloV4Out/M4p-c2-M.csv")
    YoloC3df = pd.read_csv("YoloV4Out/M4p-c3-M.csv")

    #print(list(GTdf))
    # get all GT for frame in loop
    # sort them by

    ## ------------ GET ALL GT PEOPLE ------------------- ##

    C0J = open('GroundTruth/M4p-c0-TV2.json') # C0
    C0 = json.load(C0J)
    C0 = pd.DataFrame(C0["annotations"])
    C0P1 = C0["framesP1"]  # Person 1
    C0P1 = pd.Series(C0P1[0])
    C0P2 = C0["framesP2"]  # Person 2
    C0P2 = pd.Series(C0P2[1])
    C0P3 = C0["framesP3"]  # Person 3
    C0P3 = pd.Series(C0P3[2])
    C0P4 = C0["framesP4"]  # Person 4
    C0P4 = pd.Series(C0P4[3])
    C0P4x = C0P4.head(0)

    C1J = open('GroundTruth/M4p-c1-TV2.json')  # C1
    C1 = json.load(C1J)
    C1 = pd.DataFrame(C1["annotations"])
    C1P1 = C1["framesP1"]  # Person 1
    C1P1 = pd.Series(C1P1[1])
    C1P2 = C1["framesP2"]  # Person 2
    C1P2x1 = pd.Series(C1P2[4]) # sort gap
    C1P2 = pd.Series(C1P2[3])
    C1P3 = C1["framesP3"]  # Person 3
    C1P3 = pd.Series(C1P3[0])
    C1P4 = C1["framesP4"]  # Person 4
    C1P4 = pd.Series(C1P4[2])
    C1P4x = C1P4.head(0)

    C2J = open('GroundTruth/M4p-c2-TV2.json')  # C2
    C2 = json.load(C2J)
    C2 = pd.DataFrame(C2["annotations"])
    C2P1 = C2["framesP1"]  # Person 1
    C2P1 = pd.Series(C2P1[0])
    C2P2 = C2["framesP2"]  # Person 2
    C2P2x1 = pd.Series(C2P2[3])
    C2P2 = pd.Series(C2P2[2])  # for gap
    C2P3 = C2["framesP3"]  # Person 3
    C2P3 = pd.Series(C2P3[0])
    C2P4 = C2["framesP4"]  # Person 4
    C2P4 = pd.Series(C2P4[4])
    C2P4x = C2P4.head(0)

    C3J = open('GroundTruth/M4p-c3-TV2.json')  # C3
    C3 = json.load(C3J)
    C3 = pd.DataFrame(C3["annotations"])
    C3P1 = C3["framesP1"]  # Person 1
    C3P1 = pd.Series(C3P1[4])
    C3P2 = C3["framesP2"]  # Person 2
    C3P2x1 = pd.Series(C3P2[2])
    C3P2x2 = pd.Series(C3P2[5])
    C3P2 = pd.Series(C3P2[0])
    C3P3 = C3["framesP3"]  # Person 3
    C3P3x1 = pd.Series(C3P3[3])
    C3P3x2 = pd.Series(C3P3[6])
    C3P3 = pd.Series(C3P3[1]) # three breaks in video
    C3P4 = C3["framesP4"]  # Person 4
    C3P4x1 = pd.Series(C3P4[7])
    C3P4 = pd.Series(C3P4[8])
    C3P4x = C3P4.head(0)

    ## ------------ GET ALL GT PEOPLE ------------------- ##
    #print(C0P4[3])
    #print(C0P4.head(2))

    for x in range(209): # P4 is delayed so it starts after the gap
        C0P4x = C0P4x.append(emptyRow)
    C0P4 = C0P4x.append(C0P4)
    C0P4 = C0P4.to_frame()
    C0P4 = C0P4.reset_index()
    C0P4 = pd.Series(C0P4[0])

    for x in range(141): # P4 is delayed so it starts after the gap
        C1P4x = C1P4x.append(emptyRow)
    C1P4 = C1P4x.append(C1P4)
    C1P4 = C1P4.to_frame()
    C1P4 = C1P4.reset_index()
    C1P4 = pd.Series(C1P4[0])

    for x in range(187): # P4 is delayed so it starts after the gap
        C2P4x = C2P4x.append(emptyRow)
    C2P4 = C2P4x.append(C2P4)
    C2P4 = C2P4.to_frame()
    C2P4 = C2P4.reset_index()
    C2P4 = pd.Series(C2P4[0])

    for x in range(218): # P4 is delayed so it starts after the gap
        C3P4x = C3P4x.append(emptyRow)
    C3P4 = C3P4x.append(C3P4)


    for x in range(60): # P2 in camera one is splt into 2 parts with 60 frame gap
        C1P2 = C1P2.append(emptyRow)
    C1P2 = C1P2.append(C1P2x1)
    C1P2 = C1P2.to_frame()
    C1P2 = C1P2.reset_index()
    C1P2 = pd.Series(C1P2[0])

    #print(C2P2x1)
    for x in range(50): # P2 in camera one is splt into 2 parts with 60 frame gap
        C2P2 = C2P2.append(emptyRow)
    C2P2 = C2P2.append(C2P2x1)
    C2P2 = C2P2.to_frame()
    C2P2 = C2P2.reset_index()
    C2P2 = pd.Series(C2P2[0])

    for x in range(134): # P2 in camera two is splt into 2 parts with 60 frame gap
        C3P2 = C3P2.append(emptyRow)
    C3P2 = C3P2.append(C3P2x1)

    for x in range(60): # P2 in camera one is splt into 2 parts with 60 frame gap
        C3P2 = C3P2.append(emptyRow)
    C3P2 = C3P2.append(C3P2x2)
    C3P2 = C3P2.to_frame()
    C3P2 = C3P2.reset_index()
    C3P2 = pd.Series(C3P2[0])

    for x in range(42): # P2 in camera one is splt into 2 parts with 60 frame gap
        C3P3 = C3P3.append(emptyRow)
    C3P3 = C3P3.append(C3P3x1)

    for x in range(37): # P2 in camera one is splt into 2 parts with 60 frame gap
        C3P3 = C3P3.append(emptyRow)
    C3P3 = C3P3.append(C3P3x2)
    C3P3 = C3P3.to_frame()
    C3P3 = C3P3.reset_index()
    C3P3 = pd.Series(C3P3[0])

    for x in range(56): # P2 in camera one is splt into 2 parts with 60 frame gap
        C3P4 = C3P4.append(emptyRow)
    C3P4 = C3P4.append(C3P4x1)
    C3P4 = C3P4.to_frame()
    C3P4 = C3P4.reset_index()
    C3P4 = pd.Series(C3P4[0])

    # --- Fill Gaps in Ground Truth --- #

    # for every video ## this is how it works (KEEP)
        # for every frame
            # for every person in the frame
                # for system
                    # check if object id is in system
                        # if so check if it is the correct annotation # add to correct counter
                            # then check IoU # record IoU in list to be avg
                        # if correct id is not there, find if there is object where it should be
                            # then check IoU # and record it in list of avg IoU not correct id
                        # if cannot find match, record as non found detections


    SCorrectIDNOTCorrectPersonC = 0 # correct Id and incorrect Person
    SCorrectIDCounter = 0 # correct person and ID
    SNotCorIDCorrectPersonC = 0 # incorrect id and correct person
    SNotCorIDNotPersonC = 0 # incorrect both
    SCIDCP = [] # correct ID and correct Person
    SCIDIP = [] # correct ID and incorrect Person
    SIIDCP = [] # Incorrect ID and Correct person
    SIIDIP = [] # Incorrect ID and Incorrect Person

    YCorrectIDNOTCorrectPersonC = 0 # correct Id and incorrect Person
    YCorrectIDCounter = 0 # correct person and ID
    YNotCorIDCorrectPersonC = 0 # incorrect id and correct person
    YNotCorIDNotPersonC = 0 # incorrect both

    YCIDCP = [] # correct ID and correct Person
    YCIDIP = [] # correct ID and incorrect Person
    YIIDCP = [] # Incorrect ID and Correct person
    YIIDIP = [] # Incorrect ID and Incorrect Person

    for FrameNum in range(3,900): # should be 3 to 900, but order for 4 is wrong
        #print(FrameNum)
        GTPeople = [[C0P1[FrameNum]], [C0P2[FrameNum]], [C0P3[FrameNum]], [C0P4[FrameNum]]]
        for P in GTPeople:
            #print(SystemC0df.loc[SystemC0df['frame_id'] == FrameNum])
            for S in range(len(SystemC0df.loc[SystemC0df['frame_id'] == FrameNum])):
                #print(f"S: {S}. {SystemC0df.iloc[S]['object_id']} = {P[0]['instance_id']['value']}")
                if SystemC0df.iloc[S]['object_id'] == P[0]['instance_id']['value']:
                    if (same_BB([SystemC0df.iloc[S]['xmin'], SystemC0df.iloc[S]['ymin']],[P[0]['bounding_box']['x'], P[0]['bounding_box']['y']])):
                        SCorrectIDCounter +=1
                        b1 =[[SystemC0df.iloc[S]['xmin'], SystemC0df.iloc[S]['ymin']],[SystemC0df.iloc[S]['xmax'], SystemC0df.iloc[S]['ymin']],[SystemC0df.iloc[S]['xmax'], SystemC0df.iloc[S]['ymax']],[SystemC0df.iloc[S]['xmin'], SystemC0df.iloc[S]['ymax']]]
                        b2 = [[P[0]['bounding_box']['x'],P[0]['bounding_box']['y']], [P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'], P[0]['bounding_box']['y']],[P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'], P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']],[P[0]['bounding_box']['x'], P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']]]
                        IoU = calculate_iou(b1,b2)
                        SCIDCP.append(IoU)#store here
                    else:
                        SCorrectIDNOTCorrectPersonC +=1
                        b1 = [[SystemC0df.iloc[S]['xmin'], SystemC0df.iloc[S]['ymin']],[SystemC0df.iloc[S]['xmax'], SystemC0df.iloc[S]['ymin']],[SystemC0df.iloc[S]['xmax'], SystemC0df.iloc[S]['ymax']],[SystemC0df.iloc[S]['xmin'], SystemC0df.iloc[S]['ymax']]]
                        b2 = [[P[0]['bounding_box']['x'], P[0]['bounding_box']['y']],[P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'], P[0]['bounding_box']['y']],[P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'],P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']],[P[0]['bounding_box']['x'], P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']]]
                        IoU = calculate_iou(b1, b2)
                        SCIDIP.append(IoU)# store here
                elif (same_BB([SystemC0df.iloc[S]['xmin'], SystemC0df.iloc[S]['ymin']],[P[0]['bounding_box']['x'], P[0]['bounding_box']['y']])):
                    SNotCorIDCorrectPersonC += 1
                    b1 = [[SystemC0df.iloc[S]['xmin'], SystemC0df.iloc[S]['ymin']],[SystemC0df.iloc[S]['xmax'], SystemC0df.iloc[S]['ymin']],[SystemC0df.iloc[S]['xmax'], SystemC0df.iloc[S]['ymax']],[SystemC0df.iloc[S]['xmin'], SystemC0df.iloc[S]['ymax']]]
                    b2 = [[P[0]['bounding_box']['x'], P[0]['bounding_box']['y']],[P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'], P[0]['bounding_box']['y']],[P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'],P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']],[P[0]['bounding_box']['x'], P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']]]
                    IoU = calculate_iou(b1, b2)
                    SIIDCP.append(IoU)# store here
                else :
                    SNotCorIDNotPersonC +=1
                    b1 = [[SystemC0df.iloc[S]['xmin'], SystemC0df.iloc[S]['ymin']],[SystemC0df.iloc[S]['xmax'], SystemC0df.iloc[S]['ymin']],[SystemC0df.iloc[S]['xmax'], SystemC0df.iloc[S]['ymax']],[SystemC0df.iloc[S]['xmin'], SystemC0df.iloc[S]['ymax']]]
                    b2 = [[P[0]['bounding_box']['x'], P[0]['bounding_box']['y']],[P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'], P[0]['bounding_box']['y']],[P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'],P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']],[P[0]['bounding_box']['x'], P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']]]
                    IoU = calculate_iou(b1, b2)
                    SIIDIP.append(IoU) #can add this to store complety incorrect Values
                    ## BREAK IN DATASETS -------------------------------------------------------------------------------------
            for S in range(len(YoloC0df.loc[SystemC0df['frame_id'] == FrameNum])):
                #print(SystemC0df.iloc[S]['object_id'])
                if YoloC0df.iloc[S]['object_id'] == P[0]['instance_id']['value']:
                    if (same_BB([YoloC0df.iloc[S]['xmin'], YoloC0df.iloc[S]['ymin']],[P[0]['bounding_box']['x'], P[0]['bounding_box']['y']])):
                        YCorrectIDCounter +=1
                        b1 =[[YoloC0df.iloc[S]['xmin'], YoloC0df.iloc[S]['ymin']],[YoloC0df.iloc[S]['xmax'], YoloC0df.iloc[S]['ymin']],[YoloC0df.iloc[S]['xmax'], YoloC0df.iloc[S]['ymax']],[YoloC0df.iloc[S]['xmin'], YoloC0df.iloc[S]['ymax']]]
                        b2 = [[P[0]['bounding_box']['x'],P[0]['bounding_box']['y']], [P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'], P[0]['bounding_box']['y']],[P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'], P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']],[P[0]['bounding_box']['x'], P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']]]
                        IoU = calculate_iou(b1,b2)
                        YCIDCP.append(IoU)#store here
                    else:
                        YCorrectIDNOTCorrectPersonC +=1
                        b1 = [[YoloC0df.iloc[S]['xmin'], YoloC0df.iloc[S]['ymin']],[YoloC0df.iloc[S]['xmax'], YoloC0df.iloc[S]['ymin']],[YoloC0df.iloc[S]['xmax'], YoloC0df.iloc[S]['ymax']],[YoloC0df.iloc[S]['xmin'], YoloC0df.iloc[S]['ymax']]]
                        b2 = [[P[0]['bounding_box']['x'], P[0]['bounding_box']['y']],[P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'], P[0]['bounding_box']['y']],[P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'],P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']],[P[0]['bounding_box']['x'], P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']]]
                        IoU = calculate_iou(b1, b2)
                        YCIDIP.append(IoU)# store here
                elif (same_BB([YoloC0df.iloc[S]['xmin'], YoloC0df.iloc[S]['ymin']],[P[0]['bounding_box']['x'], P[0]['bounding_box']['y']])):
                    YNotCorIDCorrectPersonC += 1
                    b1 = [[YoloC0df.iloc[S]['xmin'], YoloC0df.iloc[S]['ymin']],[YoloC0df.iloc[S]['xmax'], YoloC0df.iloc[S]['ymin']],[YoloC0df.iloc[S]['xmax'], YoloC0df.iloc[S]['ymax']],[YoloC0df.iloc[S]['xmin'], YoloC0df.iloc[S]['ymax']]]
                    b2 = [[P[0]['bounding_box']['x'], P[0]['bounding_box']['y']],[P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'], P[0]['bounding_box']['y']],[P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'],P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']],[P[0]['bounding_box']['x'], P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']]]
                    IoU = calculate_iou(b1, b2)
                    YIIDCP.append(IoU)# store here
                else:
                    YNotCorIDNotPersonC +=1
                    b1 = [[YoloC0df.iloc[S]['xmin'], YoloC0df.iloc[S]['ymin']],[YoloC0df.iloc[S]['xmax'], YoloC0df.iloc[S]['ymin']],[YoloC0df.iloc[S]['xmax'], YoloC0df.iloc[S]['ymax']],[YoloC0df.iloc[S]['xmin'], YoloC0df.iloc[S]['ymax']]]
                    b2 = [[P[0]['bounding_box']['x'], P[0]['bounding_box']['y']],[P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'], P[0]['bounding_box']['y']],[P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'],P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']],[P[0]['bounding_box']['x'], P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']]]
                    IoU = calculate_iou(b1, b2)
                    YIIDIP.append(IoU) #can add this to store complety incorrect Values

    for FrameNum in range(3, 900):  # should be 3 to 900, but order for 4 is wrong
        print(FrameNum)
        GTPeople = [[C1P1[FrameNum]], [C1P2[FrameNum]], [C1P3[FrameNum]], [C1P4[FrameNum]]]
        for P in GTPeople:
            # print(SystemC1df.loc[SystemC1df['frame_id'] == FrameNum])
            for S in range(len(SystemC1df.loc[SystemC1df['frame_id'] == FrameNum])):
                # print(f"S: {S}. {SystemC1df.iloc[S]['object_id']} = {P[0]['instance_id']['value']}")
                if SystemC1df.iloc[S]['object_id'] == P[0]['instance_id']['value']:
                    if (same_BB([SystemC1df.iloc[S]['xmin'], SystemC1df.iloc[S]['ymin']],
                                [P[0]['bounding_box']['x'], P[0]['bounding_box']['y']])):
                        SCorrectIDCounter += 1
                        b1 = [[SystemC1df.iloc[S]['xmin'], SystemC1df.iloc[S]['ymin']],
                              [SystemC1df.iloc[S]['xmax'], SystemC1df.iloc[S]['ymin']],
                              [SystemC1df.iloc[S]['xmax'], SystemC1df.iloc[S]['ymax']],
                              [SystemC1df.iloc[S]['xmin'], SystemC1df.iloc[S]['ymax']]]
                        b2 = [[P[0]['bounding_box']['x'], P[0]['bounding_box']['y']],
                              [P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'], P[0]['bounding_box']['y']],
                              [P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'],
                               P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']],
                              [P[0]['bounding_box']['x'], P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']]]
                        IoU = calculate_iou(b1, b2)
                        SCIDCP.append(IoU)  # store here
                    else:
                        SCorrectIDNOTCorrectPersonC += 1
                        b1 = [[SystemC1df.iloc[S]['xmin'], SystemC1df.iloc[S]['ymin']],
                              [SystemC1df.iloc[S]['xmax'], SystemC1df.iloc[S]['ymin']],
                              [SystemC1df.iloc[S]['xmax'], SystemC1df.iloc[S]['ymax']],
                              [SystemC1df.iloc[S]['xmin'], SystemC1df.iloc[S]['ymax']]]
                        b2 = [[P[0]['bounding_box']['x'], P[0]['bounding_box']['y']],
                              [P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'], P[0]['bounding_box']['y']],
                              [P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'],
                               P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']],
                              [P[0]['bounding_box']['x'], P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']]]
                        IoU = calculate_iou(b1, b2)
                        SCIDIP.append(IoU)  # store here
                elif (same_BB([SystemC1df.iloc[S]['xmin'], SystemC1df.iloc[S]['ymin']],
                              [P[0]['bounding_box']['x'], P[0]['bounding_box']['y']])):
                    SNotCorIDCorrectPersonC += 1
                    b1 = [[SystemC1df.iloc[S]['xmin'], SystemC1df.iloc[S]['ymin']],
                          [SystemC1df.iloc[S]['xmax'], SystemC1df.iloc[S]['ymin']],
                          [SystemC1df.iloc[S]['xmax'], SystemC1df.iloc[S]['ymax']],
                          [SystemC1df.iloc[S]['xmin'], SystemC1df.iloc[S]['ymax']]]
                    b2 = [[P[0]['bounding_box']['x'], P[0]['bounding_box']['y']],
                          [P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'], P[0]['bounding_box']['y']],
                          [P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'],
                           P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']],
                          [P[0]['bounding_box']['x'], P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']]]
                    IoU = calculate_iou(b1, b2)
                    SIIDCP.append(IoU)  # store here
                else:
                    SNotCorIDNotPersonC += 1
                    b1 = [[SystemC1df.iloc[S]['xmin'], SystemC1df.iloc[S]['ymin']],
                          [SystemC1df.iloc[S]['xmax'], SystemC1df.iloc[S]['ymin']],
                          [SystemC1df.iloc[S]['xmax'], SystemC1df.iloc[S]['ymax']],
                          [SystemC1df.iloc[S]['xmin'], SystemC1df.iloc[S]['ymax']]]
                    b2 = [[P[0]['bounding_box']['x'], P[0]['bounding_box']['y']],
                          [P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'], P[0]['bounding_box']['y']],
                          [P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'],
                           P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']],
                          [P[0]['bounding_box']['x'], P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']]]
                    IoU = calculate_iou(b1, b2)
                    SIIDIP.append(IoU)  # can add this to store complety incorrect Values
                    ## BREAK IN DATASETS -------------------------------------------------------------------------------------
            for S in range(len(YoloC1df.loc[SystemC1df['frame_id'] == FrameNum])):
                # print(SystemC1df.iloc[S]['object_id'])
                if YoloC1df.iloc[S]['object_id'] == P[0]['instance_id']['value']:
                    if (same_BB([YoloC1df.iloc[S]['xmin'], YoloC1df.iloc[S]['ymin']],
                                [P[0]['bounding_box']['x'], P[0]['bounding_box']['y']])):
                        YCorrectIDCounter += 1
                        b1 = [[YoloC1df.iloc[S]['xmin'], YoloC1df.iloc[S]['ymin']],
                              [YoloC1df.iloc[S]['xmax'], YoloC1df.iloc[S]['ymin']],
                              [YoloC1df.iloc[S]['xmax'], YoloC1df.iloc[S]['ymax']],
                              [YoloC1df.iloc[S]['xmin'], YoloC1df.iloc[S]['ymax']]]
                        b2 = [[P[0]['bounding_box']['x'], P[0]['bounding_box']['y']],
                              [P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'], P[0]['bounding_box']['y']],
                              [P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'],
                               P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']],
                              [P[0]['bounding_box']['x'], P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']]]
                        IoU = calculate_iou(b1, b2)
                        YCIDCP.append(IoU)  # store here
                    else:
                        YCorrectIDNOTCorrectPersonC += 1
                        b1 = [[YoloC1df.iloc[S]['xmin'], YoloC1df.iloc[S]['ymin']],
                              [YoloC1df.iloc[S]['xmax'], YoloC1df.iloc[S]['ymin']],
                              [YoloC1df.iloc[S]['xmax'], YoloC1df.iloc[S]['ymax']],
                              [YoloC1df.iloc[S]['xmin'], YoloC1df.iloc[S]['ymax']]]
                        b2 = [[P[0]['bounding_box']['x'], P[0]['bounding_box']['y']],
                              [P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'], P[0]['bounding_box']['y']],
                              [P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'],
                               P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']],
                              [P[0]['bounding_box']['x'], P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']]]
                        IoU = calculate_iou(b1, b2)
                        YCIDIP.append(IoU)  # store here
                elif (same_BB([YoloC1df.iloc[S]['xmin'], YoloC1df.iloc[S]['ymin']],
                              [P[0]['bounding_box']['x'], P[0]['bounding_box']['y']])):
                    YNotCorIDCorrectPersonC += 1
                    b1 = [[YoloC1df.iloc[S]['xmin'], YoloC1df.iloc[S]['ymin']],
                          [YoloC1df.iloc[S]['xmax'], YoloC1df.iloc[S]['ymin']],
                          [YoloC1df.iloc[S]['xmax'], YoloC1df.iloc[S]['ymax']],
                          [YoloC1df.iloc[S]['xmin'], YoloC1df.iloc[S]['ymax']]]
                    b2 = [[P[0]['bounding_box']['x'], P[0]['bounding_box']['y']],
                          [P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'], P[0]['bounding_box']['y']],
                          [P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'],
                           P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']],
                          [P[0]['bounding_box']['x'], P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']]]
                    IoU = calculate_iou(b1, b2)
                    YIIDCP.append(IoU)  # store here
                else:
                    YNotCorIDNotPersonC += 1
                    b1 = [[YoloC1df.iloc[S]['xmin'], YoloC1df.iloc[S]['ymin']],
                          [YoloC1df.iloc[S]['xmax'], YoloC1df.iloc[S]['ymin']],
                          [YoloC1df.iloc[S]['xmax'], YoloC1df.iloc[S]['ymax']],
                          [YoloC1df.iloc[S]['xmin'], YoloC1df.iloc[S]['ymax']]]
                    b2 = [[P[0]['bounding_box']['x'], P[0]['bounding_box']['y']],
                          [P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'], P[0]['bounding_box']['y']],
                          [P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'],
                           P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']],
                          [P[0]['bounding_box']['x'], P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']]]
                    IoU = calculate_iou(b1, b2)
                    YIIDIP.append(IoU)  # can add this to store complety incorrect Values

    for FrameNum in range(3, 900):  # should be 3 to 900, but order for 4 is wrong
        print(FrameNum)
        GTPeople = [[C3P1[FrameNum]], [C3P2[FrameNum]], [C3P3[FrameNum]], [C3P4[FrameNum]]]
        for P in GTPeople:
            # print(SystemC3df.loc[SystemC3df['frame_id'] == FrameNum])
            for S in range(len(SystemC3df.loc[SystemC3df['frame_id'] == FrameNum])):
                # print(f"S: {S}. {SystemC3df.iloc[S]['object_id']} = {P[0]['instance_id']['value']}")
                if SystemC3df.iloc[S]['object_id'] == P[0]['instance_id']['value']:
                    if (same_BB([SystemC3df.iloc[S]['xmin'], SystemC3df.iloc[S]['ymin']],
                                [P[0]['bounding_box']['x'], P[0]['bounding_box']['y']])):
                        SCorrectIDCounter += 1
                        b1 = [[SystemC3df.iloc[S]['xmin'], SystemC3df.iloc[S]['ymin']],
                              [SystemC3df.iloc[S]['xmax'], SystemC3df.iloc[S]['ymin']],
                              [SystemC3df.iloc[S]['xmax'], SystemC3df.iloc[S]['ymax']],
                              [SystemC3df.iloc[S]['xmin'], SystemC3df.iloc[S]['ymax']]]
                        b2 = [[P[0]['bounding_box']['x'], P[0]['bounding_box']['y']],
                              [P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'], P[0]['bounding_box']['y']],
                              [P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'],
                               P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']],
                              [P[0]['bounding_box']['x'], P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']]]
                        IoU = calculate_iou(b1, b2)
                        SCIDCP.append(IoU)  # store here
                    else:
                        SCorrectIDNOTCorrectPersonC += 1
                        b1 = [[SystemC3df.iloc[S]['xmin'], SystemC3df.iloc[S]['ymin']],
                              [SystemC3df.iloc[S]['xmax'], SystemC3df.iloc[S]['ymin']],
                              [SystemC3df.iloc[S]['xmax'], SystemC3df.iloc[S]['ymax']],
                              [SystemC3df.iloc[S]['xmin'], SystemC3df.iloc[S]['ymax']]]
                        b2 = [[P[0]['bounding_box']['x'], P[0]['bounding_box']['y']],
                              [P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'], P[0]['bounding_box']['y']],
                              [P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'],
                               P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']],
                              [P[0]['bounding_box']['x'], P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']]]
                        IoU = calculate_iou(b1, b2)
                        SCIDIP.append(IoU)  # store here
                elif (same_BB([SystemC3df.iloc[S]['xmin'], SystemC3df.iloc[S]['ymin']],
                              [P[0]['bounding_box']['x'], P[0]['bounding_box']['y']])):
                    SNotCorIDCorrectPersonC += 1
                    b1 = [[SystemC3df.iloc[S]['xmin'], SystemC3df.iloc[S]['ymin']],
                          [SystemC3df.iloc[S]['xmax'], SystemC3df.iloc[S]['ymin']],
                          [SystemC3df.iloc[S]['xmax'], SystemC3df.iloc[S]['ymax']],
                          [SystemC3df.iloc[S]['xmin'], SystemC3df.iloc[S]['ymax']]]
                    b2 = [[P[0]['bounding_box']['x'], P[0]['bounding_box']['y']],
                          [P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'], P[0]['bounding_box']['y']],
                          [P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'],
                           P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']],
                          [P[0]['bounding_box']['x'], P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']]]
                    IoU = calculate_iou(b1, b2)
                    SIIDCP.append(IoU)  # store here
                else:
                    SNotCorIDNotPersonC += 1
                    b1 = [[SystemC3df.iloc[S]['xmin'], SystemC3df.iloc[S]['ymin']],
                          [SystemC3df.iloc[S]['xmax'], SystemC3df.iloc[S]['ymin']],
                          [SystemC3df.iloc[S]['xmax'], SystemC3df.iloc[S]['ymax']],
                          [SystemC3df.iloc[S]['xmin'], SystemC3df.iloc[S]['ymax']]]
                    b2 = [[P[0]['bounding_box']['x'], P[0]['bounding_box']['y']],
                          [P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'], P[0]['bounding_box']['y']],
                          [P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'],
                           P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']],
                          [P[0]['bounding_box']['x'], P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']]]
                    IoU = calculate_iou(b1, b2)
                    SIIDIP.append(IoU)  # can add this to store complety incorrect Values
                    ## BREAK IN DATASETS -------------------------------------------------------------------------------------
            for S in range(len(YoloC3df.loc[SystemC3df['frame_id'] == FrameNum])):
                # print(SystemC3df.iloc[S]['object_id'])
                if YoloC3df.iloc[S]['object_id'] == P[0]['instance_id']['value']:
                    if (same_BB([YoloC3df.iloc[S]['xmin'], YoloC3df.iloc[S]['ymin']],
                                [P[0]['bounding_box']['x'], P[0]['bounding_box']['y']])):
                        YCorrectIDCounter += 1
                        b1 = [[YoloC3df.iloc[S]['xmin'], YoloC3df.iloc[S]['ymin']],
                              [YoloC3df.iloc[S]['xmax'], YoloC3df.iloc[S]['ymin']],
                              [YoloC3df.iloc[S]['xmax'], YoloC3df.iloc[S]['ymax']],
                              [YoloC3df.iloc[S]['xmin'], YoloC3df.iloc[S]['ymax']]]
                        b2 = [[P[0]['bounding_box']['x'], P[0]['bounding_box']['y']],
                              [P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'], P[0]['bounding_box']['y']],
                              [P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'],
                               P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']],
                              [P[0]['bounding_box']['x'], P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']]]
                        IoU = calculate_iou(b1, b2)
                        YCIDCP.append(IoU)  # store here
                    else:
                        YCorrectIDNOTCorrectPersonC += 1
                        b1 = [[YoloC3df.iloc[S]['xmin'], YoloC3df.iloc[S]['ymin']],
                              [YoloC3df.iloc[S]['xmax'], YoloC3df.iloc[S]['ymin']],
                              [YoloC3df.iloc[S]['xmax'], YoloC3df.iloc[S]['ymax']],
                              [YoloC3df.iloc[S]['xmin'], YoloC3df.iloc[S]['ymax']]]
                        b2 = [[P[0]['bounding_box']['x'], P[0]['bounding_box']['y']],
                              [P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'], P[0]['bounding_box']['y']],
                              [P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'],
                               P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']],
                              [P[0]['bounding_box']['x'], P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']]]
                        IoU = calculate_iou(b1, b2)
                        YCIDIP.append(IoU)  # store here
                elif (same_BB([YoloC3df.iloc[S]['xmin'], YoloC3df.iloc[S]['ymin']],
                              [P[0]['bounding_box']['x'], P[0]['bounding_box']['y']])):
                    YNotCorIDCorrectPersonC += 1
                    b1 = [[YoloC3df.iloc[S]['xmin'], YoloC3df.iloc[S]['ymin']],
                          [YoloC3df.iloc[S]['xmax'], YoloC3df.iloc[S]['ymin']],
                          [YoloC3df.iloc[S]['xmax'], YoloC3df.iloc[S]['ymax']],
                          [YoloC3df.iloc[S]['xmin'], YoloC3df.iloc[S]['ymax']]]
                    b2 = [[P[0]['bounding_box']['x'], P[0]['bounding_box']['y']],
                          [P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'], P[0]['bounding_box']['y']],
                          [P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'],
                           P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']],
                          [P[0]['bounding_box']['x'], P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']]]
                    IoU = calculate_iou(b1, b2)
                    YIIDCP.append(IoU)  # store here
                else:
                    YNotCorIDNotPersonC += 1
                    b1 = [[YoloC3df.iloc[S]['xmin'], YoloC3df.iloc[S]['ymin']],
                          [YoloC3df.iloc[S]['xmax'], YoloC3df.iloc[S]['ymin']],
                          [YoloC3df.iloc[S]['xmax'], YoloC3df.iloc[S]['ymax']],
                          [YoloC3df.iloc[S]['xmin'], YoloC3df.iloc[S]['ymax']]]
                    b2 = [[P[0]['bounding_box']['x'], P[0]['bounding_box']['y']],
                          [P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'], P[0]['bounding_box']['y']],
                          [P[0]['bounding_box']['x'] + P[0]['bounding_box']['h'],
                           P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']],
                          [P[0]['bounding_box']['x'], P[0]['bounding_box']['y'] + P[0]['bounding_box']['w']]]
                    IoU = calculate_iou(b1, b2)
                    YIIDIP.append(IoU)  # can add this to store complety incorrect Values
    #fin eval checks

    # print out systems IoU, correct deterctions
    # print out yolos IoU and correct detections

    # result print outs
    print("Statistics for System results")
    print(f"Number of correct ID of correct Person found, in all frames: {len(SCIDCP)}")
    print(f"Number of correct ID but incorrect person found, in all frames: {len(SCIDIP)}")
    print(f"Number of incorrect ID but correct person found, in all frames: {len(SIIDCP)}")
    print(f"Number of incorrect ID and person correct found, in all frames: {len(SIIDIP)}")
    #print(SCIDCP)
    print("")
    print("Statistics for Just Yolov4 tracking results")
    print(f"Number of correct ID of correct Person found, in all frames: {len(YCIDCP)}")
    print(f"Number of correct ID but incorrect person found, in all frames: {len(YCIDIP)}")
    print(f"Number of incorrect ID but correct person found, in all frames: {len(YIIDCP)}")
    print(f"Number of incorrect ID and person correct found, in all frames: {len(YIIDIP)}")
    #print(YCIDCP)
