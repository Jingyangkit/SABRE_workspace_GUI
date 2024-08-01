def Bubbling(state):
    if state == 0:
        print('Bubbling open')
    else:
        print('Bubbling close')

def Gripper(state):
    if state == 0:
        print('Gripper open')
    else:
        print('Gripper close')

def State_of_the_Cooling_workspace(state):
    if state == 0:
        print('work')
    else:
        print('do not work')
def Experiment_start():
    print('Experiment_start')

def emergency_stop():
    print('emergency_stop')

#这个函数可以判断四个小圆圈哪个被选中
def four_cycle(state):
    if state == "Temp_Field_Sweeping":
        cycle = "TFS"
    elif state == "Temp_Balance_Time":
        cycle = "TBT"
    elif state == "Solidification_Buildup_Time":
        cycle = "SBT"
    elif state == "Temp_Alternation":
        cycle = "TA"
    else:
        cycle = "nothing"
    return cycle

def Magnetic_Field_Sweeping_parameter_Set(Magnetic_Field_1st_value, Magnetic_Field_2nd_value, Time_Delta, Bubbling_Time, Wating_time):
    print("Magnetic_Field_1st_value is [%s]" % Magnetic_Field_1st_value)
    print("Magnetic_Field_2nd_value is [%s]" % Magnetic_Field_2nd_value)
    print("Time_Delta is [%s]" % Time_Delta)
    print("Bubbling_Time is [%s]" % Bubbling_Time)
    print("Wating_time is [%s]" % Wating_time)

def Polarization_Relaxation_Time_parameter_Set(Low_Field, High_Field, Bubbling_Time, Wating_Time):
    print("Low_Field is [%s]" % Low_Field)
    print("High_Field is [%s]" % High_Field)
    print("Bubbling_Time is [%s]" % Bubbling_Time)
    print("Wating_Time is [%s]" % Wating_Time)

def Polarization_Buildup_Time_parameter_Set(Bubbling_1st_value, Bubbling_Time_2nd_value, Time_Delta, Magnetic_Field, Wating_Time):
    print("Bubbling_1st_value is [%s]" % Bubbling_1st_value)
    print("Bubbling_Time_2nd_value is [%s]" % Bubbling_Time_2nd_value)
    print("Time_Delta is [%s]" % Time_Delta)
    print("Magnetic_Field is [%s]" % Magnetic_Field)
    print("Wating_Time is [%s]" % Wating_Time)

def Pulse_Sequence_parameter_Set(Low_Field, LSweeping_Time_1st_value, LSweeping_Time_2nd_value, LSweeping_Time_Delta, High_Field, HSweeping_Time_1st_value, HSweeping_Time_2nd_value, HSweeping_Time_Delta, Bubbling_Time):
    print("Low_Field is [%s]" % Low_Field)
    print("LSweeping_Time_1st_value is [%s]" % LSweeping_Time_1st_value)
    print("LSweeping_Time_2nd_value is [%s]" % LSweeping_Time_2nd_value)
    print("LSweeping_Time_Delta is [%s]" % LSweeping_Time_Delta)
    print("High_Field is [%s]" % High_Field)
    print("HSweeping_Time_1st_value is [%s]" % HSweeping_Time_1st_value)
    print("HSweeping_Time_2nd_value is [%s]" % HSweeping_Time_2nd_value)
    print("HSweeping_Time_Delta is [%s]" % HSweeping_Time_Delta)
    print("Bubbling_Time is [%s]" % Bubbling_Time)



