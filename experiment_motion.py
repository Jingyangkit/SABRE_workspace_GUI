import time
import sys
import numpy as np
from SABRE_Motion import RoboticArmMotion
from SABRE_SerialCom import Arduino
from newFeedback import feedback
from AcquireCustom import shimming
MYPATH = 'C:/Users/Magritek/Documents/Moritz/'
DATAPATH = 'C:/Users/Magritek/Documents/Moritz/data/'
sys.path.append(MYPATH)
import utils_Spinsolve

MagneticFieldValue = [97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122]
Finished = 0
Measure_Time = 6 # stay in Spinsolve

distortions = shimming.distortions


def magnetic_field_sweeping(self, magnetic_field_1st, magnetic_field_2nd, delta, wating_time_inSS, bubbling_time):

    com = utils_Spinsolve.init(gui=True)
    magnetic_field = magnetic_field_1st
    shimming_times = 0
    time.sleep(6)
    RoboticArmMotion.initialization(1)

    print('spinsove is enabled')

    Arduino.SerialComTest(1)
    time.sleep(1)
    Arduino.GrabTask_Open(1)
    RoboticArmMotion.MoveToObject(1)
    time.sleep(1)
    RoboticArmMotion.MovingObjects(1)
    time.sleep(1)
    Arduino.GrabTask_Close(1)
    time.sleep(1)

    while magnetic_field_1st != magnetic_field_2nd:
        RoboticArmMotion.MoveToMagField(1)
        time.sleep(wating_time_inSS)

        Arduino.Bubbling_Open(1, 0)
        Arduino.Start_MagneticField(1, magnetic_field, bubbling_time)

        Arduino.Bubbling_Close(1)
        Arduino.Stop_MagneticField(1)

        time.sleep(0.5)

        RoboticArmMotion.MovingToSpinsolve(1)
        time.sleep(Measure_Time)

        xaxis, initial_spectrum, fid, ref_shims = utils_Spinsolve.setShimsAndRunV3(com,
                                                                                   shimming_times + 1,
                                                                                   distortions[shimming_times],
                                                                                   return_shimvalues=True,
                                                                                   return_fid=True)
        shimming_times = shimming_times + 1

        magnetic_field = magnetic_field + delta

        if magnetic_field_1st == magnetic_field_2nd:
            print("Experiment finished")
            Arduino.GrabTask_Open(1)
            utils_Spinsolve.shutdown(com)
            RoboticArmMotion.return_Function(1)


# def Polarization_Relaxation_Time(self, magnetic_field_1st, magnetic_field_2nd, delta, wating_time_inSS, bubbling_time):
#
#     com = utils_Spinsolve.init(gui=True)
#     magnetic_field = magnetic_field_1st
#     shimming_times = 0
#     time.sleep(6)
#     RoboticArmMotion.initialization(1)
#
#     print('spinsove is enabled')
#
#     Arduino.SerialComTest(1)
#     time.sleep(1)
#     Arduino.GrabTask_Open(1)
#     RoboticArmMotion.MoveToObject(1)
#     time.sleep(1)
#     RoboticArmMotion.MovingObjects(1)
#     time.sleep(1)
#     Arduino.GrabTask_Close(1)
#     time.sleep(1)
#
#     while magnetic_field_1st != magnetic_field_2nd:
#         RoboticArmMotion.MoveToMagField(1)
#         time.sleep(wating_time_inSS)
#
#         Arduino.Bubbling_Open(1, 0)
#         Arduino.Start_MagneticField(1, magnetic_field, bubbling_time)
#
#         Arduino.Bubbling_Close(1)
#         Arduino.Stop_MagneticField(1)
#
#         time.sleep(0.5)
#
#         RoboticArmMotion.MovingToSpinsolve(1)
#         time.sleep(Measure_Time)
#
#         xaxis, initial_spectrum, fid, ref_shims = utils_Spinsolve.setShimsAndRunV3(com,
#                                                                                    shimming_times + 1,
#                                                                                    distortions[shimming_times],
#                                                                                    return_shimvalues=True,
#                                                                                    return_fid=True)
#         shimming_times = shimming_times + 1
#
#         magnetic_field = magnetic_field + delta
#
#         if magnetic_field_1st == magnetic_field_2nd:
#             print("Experiment finished")
#             Arduino.GrabTask_Open(1)
#             utils_Spinsolve.shutdown(com)
#             RoboticArmMotion.return_Function(1)

def Polarization_Buildup_Time(self, Bubbling_1st_value, Bubbling_Time_2nd_value, Time_Delta, Magnetic_Field, Wating_Time,):

    com = utils_Spinsolve.init(gui=True)
    bubbling_time = Bubbling_1st_value
    shimming_times = 0
    time.sleep(6)
    RoboticArmMotion.initialization(1)

    print('spinsove is enabled')

    Arduino.SerialComTest(1)
    time.sleep(1)
    Arduino.GrabTask_Open(1)
    RoboticArmMotion.MoveToObject(1)
    time.sleep(1)
    RoboticArmMotion.MovingObjects(1)
    time.sleep(1)
    Arduino.GrabTask_Close(1)
    time.sleep(1)

    while Bubbling_1st_value != Bubbling_Time_2nd_value:
        RoboticArmMotion.MoveToMagField(1)
        time.sleep(Wating_Time)

        Arduino.Bubbling_Open(1, 0)
        Arduino.Start_MagneticField(1, Magnetic_Field, bubbling_time)

        Arduino.Bubbling_Close(1)
        Arduino.Stop_MagneticField(1)

        time.sleep(0.5)

        RoboticArmMotion.MovingToSpinsolve(1)
        time.sleep(Measure_Time)

        xaxis, initial_spectrum, fid, ref_shims = utils_Spinsolve.setShimsAndRunV3(com,
                                                                                   shimming_times + 1,
                                                                                   distortions[shimming_times],
                                                                                   return_shimvalues=True,
                                                                                   return_fid=True)
        shimming_times = shimming_times + 1

        bubbling_time = bubbling_time + Time_Delta

        if Bubbling_1st_value == Bubbling_Time_2nd_value:
            print("Experiment finished")
            Arduino.GrabTask_Open(1)
            utils_Spinsolve.shutdown(com)
            RoboticArmMotion.return_Function(1)


def Pulse_Sequence(self, Low_Field, LSweeping_Time_1st_value, LSweeping_Time_2nd_value, LSweeping_Time_Delta, High_Field, HSweeping_Time_1st_value, HSweeping_Time_2nd_value, HSweeping_Time_Delta, Bubbling_Time):

    com = utils_Spinsolve.init(gui=True)
    if LSweeping_Time_Delta == 0 and HSweeping_Time_Delta != 0:
        delta= HSweeping_Time_Delta
        time_1 = HSweeping_Time_1st_value
        time_2 = HSweeping_Time_2nd_value
        field = High_Field
    elif LSweeping_Time_Delta != 0 and HSweeping_Time_Delta == 0:
        delta = LSweeping_Time_Delta
        time_1 = LSweeping_Time_1st_value
        time_2 = LSweeping_Time_2nd_value
        field = Low_Field
    else:
        delta = 0
        time_1 = 0
        time_2 = 0
        field = 0

    shimming_times = 0
    time.sleep(6)
    RoboticArmMotion.initialization(1)

    print('spinsove is enabled')

    Arduino.SerialComTest(1)
    time.sleep(1)
    Arduino.GrabTask_Open(1)
    RoboticArmMotion.MoveToObject(1)
    time.sleep(1)
    RoboticArmMotion.MovingObjects(1)
    time.sleep(1)
    Arduino.GrabTask_Close(1)
    time.sleep(1)

    while time_1 != time_2:
        RoboticArmMotion.MoveToMagField(1)
        time.sleep(60)

        Arduino.Bubbling_Open(1, 0)
        Arduino.Start_MagneticField(1, field, time_1)

        Arduino.Bubbling_Close(1)
        Arduino.Stop_MagneticField(1)

        time.sleep(0.5)

        RoboticArmMotion.MovingToSpinsolve(1)
        time.sleep(Measure_Time)

        xaxis, initial_spectrum, fid, ref_shims = utils_Spinsolve.setShimsAndRunV3(com,
                                                                                   shimming_times + 1,
                                                                                   distortions[shimming_times],
                                                                                   return_shimvalues=True,
                                                                                   return_fid=True)
        shimming_times = shimming_times + 1

        time_1 = time_1 + delta

        if time_1 == time_2:
            print("Experiment finished")
            Arduino.GrabTask_Open(1)
            utils_Spinsolve.shutdown(com)
            RoboticArmMotion.return_Function(1)



