import threading
from dobot_api import DobotApiDashboard, DobotApi, DobotApiMove, MyType
from time import sleep
import numpy as np
from newFeedback import feedback
from newSend import LJ29999, LJ30003
from threading import Thread
import time
# 全局变量(当前坐标)
current_pose = [0,0,0,0]
arrive_state = False

port29999 = ['EnableRobot', 'DisableRobot', 'ClearError', 'ResetRobot', 'SpeedFactor', 'User',
             'Tool', "RobotMode", "PayLoad", "DO", "DOExecute", 'ToolDO,ToolDOExecute', 'AO'
    , 'AOExecute', 'AccJ', 'AccL', 'SpeedJ', 'SpeedL', 'Arch', 'CP', 'SetArmOrientation'
    , 'PowerOn', 'RunScript', 'StopScript', 'PauseScript', 'ContinueScript', 'SetSafeSkin'
    , 'GetTraceStartPose', 'GetPathStartPose', 'PositiveSolution', 'InverseSolution'
    , 'SetCollisionLevel', 'HandleTrajPoints', 'GetSixForceData', 'GetAngle', 'GetPose'
    , 'EmergencyStop', 'ModbusCreate', 'ModbusClose', 'GetInBits', 'GetInRegs', 'GetCoils'
    , 'etCoils', 'GetHoldRegs', 'SetHoldRegs', 'GetErrorID', 'DI', 'ToolDI', 'AI', 'ToolAI'
    , 'DIGroup', 'DOGroup', 'BrakeControl', 'StartDrag', 'StopDrag', 'SetCollideDrag'
    , 'SetTerminalKeys', 'SetTerminal485', 'GetTerminal485', 'LoadSwitch']

port30003 = ['MovJ', 'MovL', 'JointMovJ', 'MovLIO', 'MovJIO', 'Arc', 'ServoJ'
    , 'ServoP', 'MoveJog', 'StartTrace', 'StartPath', 'StartFCTrace'
    , 'Sync', 'RelMovJTool', 'RelMovLTool', 'RelMovJUser', 'RelMovLUser'
    , 'RelJointMovJ']

def wait_arrive(point_list):
    global current_pose
    global arrive_state
    while True:
        arrive_state = True
        if current_pose is not None:
            for index in range(4):
                if (abs(current_pose[index] - point_list[index]) > 1):
                    arrive_state = False

            if arrive_state:
                #print(arrive_state)
                return

        sleep(0.001)

def FK():
    global current_pose
    fk30004 = feedback("192.168.1.6", 30004)
    while True:
        try:
            time.sleep(0.001)
            fk30004.read()
            current_pose = (fk30004.basics[3])[0:4]   #获取末端执行器坐标
            #print(current_pose)
            #print(len(current_pose))
        except:
            break

class RoboticArmMotion:

    def initialization(self):
        lj29999 = LJ29999("192.168.1.6", 29999)  # 创建实例对象
        lj30003 = LJ30003("192.168.1.6", 30003)
        thread = Thread(
            target=FK,   # 新建一个线程做数据刷新
            daemon=True  # 设置新线程为daemon线程
        )
        thread.start()

        lj29999.send("EnableRobot(0.2,16,0,0)")
        lj30003.send("LoadSwitch(1)")
        #lj30003.send("PayLoad(0.23,0.000828)")
        #lj30003.send("SetPayload(0.2)")
        lj30003.send("SpeedL(10)")
        lj30003.send("MovL(160.85, -166.53, 116, -40.73)")
        wait_arrive([160.85, -166.53, 116, -40.73])
        

    def MoveToObject(self):
        lj29999 = LJ29999("192.168.1.6", 29999)  # 创建实例对象
        lj30003 = LJ30003("192.168.1.6", 30003)
        thread = Thread(
            target=FK,  # 新建一个线程做数据刷新
            daemon=True  # 设置新线程为daemon线程
        )
        thread.start()

        lj29999.send("EnableRobot(0.2,16,0,0)")
        lj30003.send("LoadSwitch(1)")
        # lj30003.send("PayLoad(0.23,0.000828)")
        # lj30003.send("SetPayload(0.2)")
        lj30003.send("SpeedL(10)")
        lj30003.send("MovL(160.85, -166.53, 76, -38.35)")
        wait_arrive([160.85, -166.53, 76, -38.35])

        lj30003.send("MovL(219.24, -166.29, -205.16, -38.35)")
        wait_arrive([219.24, -166.29, -205.16, -38.35])

    def MoveToMagField(self):
        lj29999 = LJ29999("192.168.1.6", 29999)  # 创建实例对象
        lj30003 = LJ30003("192.168.1.6", 30003)
        thread = Thread(
            target=FK,  # 新建一个线程做数据刷新
            daemon=True  # 设置新线程为daemon线程
        )
        thread.start()

        lj29999.send("EnableRobot(0.2,17,0,0)")
        lj30003.send("LoadSwitch(1)")
        #lj30003.send("PayLoad(0.23,0.000828)")
        #lj30003.send("SetPayload(0.2)")
        lj30003.send("SpeedL(11)")
        lj30003.send("MovL(233.37, -188.22, -140.26, -38.35)")
        wait_arrive([233.37, -188.22, -140.26, -38.35])
        #time.sleep(600)
        lj30003.send("MovL(231.37, -188.2, -60.26, -38.35)")
        wait_arrive([231.37, -188.2, -60.26, -38.35])

        lj30003.send("MovL(231.56, -189.02, 90, -38.35)")
        wait_arrive([231.56, -189.02,90, -38.35])
        #time.sleep(600)
        lj30003.send("SpeedL(11)")
        #lj29999.send("EnableRobot(0.24,19,0,0)")
        lj29999.send("EnableRobot(0.23,18,0,0)")
        lj30003.send("LoadSwitch(1)")
        ########### ut磁场
        lj30003.send("MovL(-141.74, -308, 100, -116.78)")
        wait_arrive([-141.74, -308, 100, -116.78])
        lj30003.send("MovL(-141.74, -308, -126.18, -116.78)")
        wait_arrive([-141.74, -308, -126.18, -116.78])

        #lj30003.send("MovL(67.17, -251.87, 107.84, -116.78)")
        #wait_arrive([67.17, -251.87, 107.84, -116.78])
        #lj30003.send("MovL(67.17, -251.87, 105.84, -116.78)")
        #wait_arrive([67.17, -251.87, 105.84, -116.78])

        ########### mt磁场
        # lj30003.send("MovL(219.58, 208.9, 160, 46.21)")
        # wait_arrive([219.58, 208.9, 160, 46.21])
        # lj30003.send("MovL(219.58, 208.9, 15.39, 46.21)")
        # wait_arrive([219.58, 208.9, 15.39, 46.21])

    def MovingObjects(self):
        lj29999 = LJ29999("192.168.1.6", 29999)  # 创建实例对象
        lj30003 = LJ30003("192.168.1.6", 30003)
        thread = Thread(
            target=FK,  # 新建一个线程做数据刷新
            daemon=True  # 设置新线程为daemon线程
        )
        thread.start()

        lj29999.send("EnableRobot(0.2,16,0,0)")
        lj30003.send("LoadSwitch(1)")
        # lj30003.send("PayLoad(0.23,0.000828)")
        # lj30003.send("SetPayload(0.2)")
        lj30003.send("SpeedL(5)")
        lj30003.send("MovL(234.28, -189.48, -205.20, -38.35)")#last point
        wait_arrive([234.28, -189.48, -205.20, -38.35])#last point
        time.sleep(0.01)

    def MovingToSpinsolve(self):
        lj29999 = LJ29999("192.168.1.6", 29999)  # 创建实例对象
        lj30003 = LJ30003("192.168.1.6", 30003)
        thread = Thread(
            target=FK,  # 新建一个线程做数据刷新
            daemon=True  # 设置新线程为daemon线程
        )
        thread.start()

        #lj29999.send("EnableRobot(0.23,18,0,0)")
        #lj30003.send("LoadSwitch(1)")
        ## lj30003.send("PayLoad(0.23,0.000828)")
        ## lj30003.send("SetPayload(0.2)")
        lj30003.send("SpeedL(96)")
        ########### ut磁场
        lj30003.send("MovL(-141.74, -308, 100, -116.78)")
        wait_arrive([-141.74, -308, 100, -116.78])
        #lj30003.send("MovL(67.17, -251.87, 107.84, -116.78)")
        #wait_arrive([67.17, -251.87, 107.84, -116.78])


        ########### mt磁场
        # lj30003.send("MovL(219.58, 208.9, 160, 46.21)")
        # wait_arrive([219.58, 208.9, 160, 46.21])

        # put into the bore of Spinsolve for measurment
        lj30003.send("SpeedL(45)")  #(put down speed 45)
        lj30003.send("MovL(231.56, -189.02, 100, -38.35)")
        wait_arrive([231.56, -189.02, 100, -38.35])
        lj30003.send("SpeedL(23)")  #put into the bore speed 23
        lj30003.send("MovL(231.37, -188.2, -60.26, -38.35)")
        wait_arrive([231.37, -188.2, -60.26, -38.35])
        lj30003.send("MovL(233.37, -188.22, -140.26, -38.35)")
        wait_arrive([233.37, -188.22, -140.26, -38.35])
        lj30003.send("MovL(234.28, -189.48, -205.20, -38.35)") #last point
        wait_arrive([234.28, -189.48, -205.20, -38.35]) #last point
        time.sleep(0.01)
        lj29999.send("EnableRobot(0.2,16,0,0)")
        lj30003.send("LoadSwitch(1)")

    def return_Function(self):
        lj29999 = LJ29999("192.168.1.6", 29999)  # 创建实例对象
        lj30003 = LJ30003("192.168.1.6", 30003)
        thread = Thread(
            target=FK,  # 新建一个线程做数据刷新
            daemon=True  # 设置新线程为daemon线程
        )
        thread.start()

        lj29999.send("EnableRobot(0.2,16,0,0)")
        lj30003.send("LoadSwitch(1)")
        # lj30003.send("PayLoad(0.23,0.000828)")
        # lj30003.send("SetPayload(0.2)")
        lj30003.send("SpeedL(6)")
        lj30003.send("MovL(219.24, -166.29, -205.16, -40.73)")
        wait_arrive([219.24, -166.29, -205.16, -40.73])
        time.sleep(0.01)

    def disable(self):
        lj29999 = LJ29999("192.168.1.6", 29999)  # 创建实例对象
        lj30003 = LJ30003("192.168.1.6", 30003)
        thread = Thread(
            target=FK,  # 新建一个线程做数据刷新
            daemon=True  # 设置新线程为daemon线程
        )
        thread.start()

        lj29999.send("DisableRobot()")

