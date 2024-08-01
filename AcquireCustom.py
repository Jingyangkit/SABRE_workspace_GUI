import os
import sys
import json
import glob
import pickle
import torch
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nmrglue as ng
from datetime import datetime
from torch import nn
import torch.nn.functional as F
#from sklearn import svm
from sklearn.metrics import mean_absolute_error
from scipy import signal, optimize
from numpy.polynomial.polynomial import polyfit
MYPATH = 'C:/Users/Magritek/Documents/Moritz/'
DATAPATH = 'C:/Users/Magritek/Documents/Moritz/data/'
sys.path.append(MYPATH)
import utils_Spinsolve
class shimming:
    distortions = np.array([    [-57,0,0,0],
                                [-57,0,-10,0],
                                [-56,0,-20,0],
                                [-54,0,-30,0],
                                [-52,0,-39,0],
                                [-49,0,-48,0],
                                [-46,0,-57,0],
                                [-42,0,-65,0],
                                [-38,0,-72,0],
                                [-34,0,-78,0],
                                [-29,0,-84,0],
                                [-23,0,-89,0],
                                [-18,0,-92,0],
                                [-12,0,-95,0],
                                [-6,0,-96,0],
                                [0,0,-97,0],
                                [6,0,-96,0],
                                [12,0,-95,0],
                                [18,0,-92,0],
                                [23,0,-89,0],
                                [29,0,-84,0],
                                [34,0,-78,0],
                                [38,0,-72,0],
                                [42,0,-65,0],
                                [46,0,-57,0],
                                [49,0,-48,0],
                                [52,0,-39,0],
                                [54,0,-30,0],
                                [56,0,-20,0],
                                [57,0,-10,0],
                                [57,0,0,0], ])

import os

import time
from SABRE_Motion import RoboticArmMotion
from SABRE_SerialCom import Arduino
from newFeedback import feedback

#MagneticFieldValue = [97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122]
#Finished = 0
#TimeInSS = 3 # Time in Shielding system
#TimeBubbling = 3

#com = utils_Spinsolve.init(gui=True)




#RoboticArmMotion.initialization(1)
#Arduino.SerialComTest(1)
#time.sleep(1)
#Arduino.GrabTask_Open(1)
#RoboticArmMotion.MoveToObject(1)
#time.sleep(1)
#RoboticArmMotion.MovingObjects(1)
#time.sleep(1)
#Arduino.GrabTask_Close(1)
#time.sleep(1)


#for i in range(len(distortions)):
    #print(i)
    #RoboticArmMotion.MoveToMagField(1)
    #time.sleep(TimeInSS)
    #Arduino.Bubbling_Open(1, TimeBubbling)
    #Arduino.Bubbling_Close(1)
    #RoboticArmMotion.MovingToSpinsolve(1)
    #distortion = distortions[i]
    #xaxis, initial_spectrum, fid, ref_shims = utils_Spinsolve.setShimsAndRunV3(com, i+1, distortion,
                                                #return_shimvalues=True, return_fid=True)
    
    #time.sleep(2)


#Arduino.GrabTask_Open(1)
#time.sleep(2)
#RoboticArmMotion.return_Function(1)
#utils_Spinsolve.shutdown(com)

