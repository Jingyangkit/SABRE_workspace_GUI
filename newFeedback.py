import socket
import numpy as np

MyType = np.dtype([('len',np.int64,), ('digital_input_bits',np.uint64,), ('digital_output_bits',
    np.uint64,), ('robot_mode',np.uint64,), ('time_stamp',np.uint64,), ( 'time_stamp_reserve_bit', np.uint64,),
    ('test_value',np.uint64,), ('test_value_keep_bit', np.float64,), ('speed_scaling',np.float64,), ('linear_momentum_norm',np.float64,),
    ( 'v_main',np.float64,), ('v_robot',np.float64,), ('i_robot',np.float64,), ('i_robot_keep_bit1',np.float64,), ( 'i_robot_keep_bit2',np.float64,),
    ('tool_accelerometer_values', np.float64, (3, )),
    ('elbow_position', np.float64, (3, )),
    ('elbow_velocity', np.float64, (3, )),
    ('q_target', np.float64, (6, )),
    ('qd_target', np.float64, (6, )),
    ('qdd_target', np.float64, (6, )),
    ('i_target', np.float64, (6, )),
    ('m_target', np.float64, (6, )),
    ('q_actual', np.float64, (6, )),
    ('qd_actual', np.float64, (6, )),
    ('i_actual', np.float64, (6, )),
    ('actual_TCP_force', np.float64, (6, )),
    ('tool_vector_actual', np.float64, (6, )),
    ('TCP_speed_actual', np.float64, (6, )),
    ('TCP_force', np.float64, (6, )),
    ('Tool_vector_target', np.float64, (6, )),
    ('TCP_speed_target', np.float64, (6, )),
    ('motor_temperatures', np.float64, (6, )),
    ('joint_modes', np.float64, (6, )),
    ('v_actual', np.float64, (6, )),
    ('hand_type', np.int8, (4,)),
    ('user', np.int8,),
    ('tool', np.int8,),
    ('run_queued_cmd', np.int8,),
    ('pause_cmd_flag', np.int8,),
    ('velocity_ratio', np.int8,),
    ('acceleration_ratio', np.int8,),
    ('jerk_ratio', np.int8,),
    ('xyz_velocity_ratio', np.int8,),
    ('r_velocity_ratio', np.int8,),
    ('xyz_acceleration_ratio', np.int8,),
    ('r_acceleration_ratio', np.int8,),
    ('xyz_jerk_ratio', np.int8,),
    ('r_jerk_ratio', np.int8,),
    ('brake_status', np.int8,),
    ('enable_status', np.int8,),
    ('drag_status', np.int8,),
    ('running_status', np.int8,),
    ('error_status',np.int8,),
    ('jog_status', np.int8,),
    ('robot_type', np.int8,),
    ('drag_button_signal', np.int8,),
    ('enable_button_signal', np.int8,),
    ('record_button_signal', np.int8,),
    ('reappear_button_signal', np.int8,),
    ('jaw_button_signal', np.int8,),
    ('six_force_online', np.int8,),
    ('reserve2', np.int8, (82,)),
    ('m_actual', np.float64, (6,)),
    ('load', np.float64,),
    ('center_x', np.float64,),
    ('center_y', np.float64,),
    ('center_z', np.float64,),
    ('user1', np.float64, (6,)),
    ('Tool1', np.float64, (6,)),
    ('trace_index', np.float64,),
    ('six_force_value', np.float64, (6,)),
    ('target_quaternion', np.float64, (4,)),
    ('actual_quaternion', np.float64, (4,)),
    ('reserve3',np.int8, (24,))
     ])

i = 0
class feedback:
    def __init__(self, ip, port):
        self.socket_feedback = 0
        if port == 30004:              #判断端口号是否为30004
            try:    #异常判断
                #创建套接字
                self.socket_feedback = socket.socket()
                #连接
                self.socket_feedback.connect((ip, port))
            except:
                print("连接失败")
        else:
            print("端口输入错误!")
    def read(self):
        try:
            self.all = self.socket_feedback.recv(10240)
            data = self.all[0:1440]
            np.frombuffer(data[624:672], dtype=(np.float64, (6,)))  #读取624—672里面有六个数据每个后面加，
            a = np.frombuffer(data, dtype=MyType)        #将收到的数据转换为上面定义的类型
            #print((a['qd_actual'])[0])
            num1 = "{:.4f}".format((a['tool_vector_actual'])[0][0])
            num2 = "{:.4f}".format((a['tool_vector_actual'])[0][1])
            num3 = "{:.4f}".format((a['tool_vector_actual'])[0][2])
            num4 = "{:.4f}".format((a['tool_vector_actual'])[0][3])
            #print('['+ num1 + ',' + num2 + ',' + num3 + ','+ num4 + ']'+ ',', end='')
            if hex((a['test_value'][0])) == '0x123456789abcdef':  #判断数据正确取需要的值
                self.basics = ('TCP笛卡尔实际坐标值:', (a['tool_vector_actual'])[0],
                                 '笛卡尔目标坐标值:', (a['Tool_vector_target'])[0],
                                 '关节实际位置:', (a['q_actual'])[0],
                                 '实际关节电流:', (a['i_actual'])[0],
                                 ' 控制板电压:', (a['v_main'])[0],
                                 ' 机器人电压:', (a['v_robot'])[0],
                                 ' 机器人电流:', (a['i_robot'])[0],)
        except:
            return "读取错误"

    def close(self): #关闭连接
        if (self.socket_feedback != 0):
            self.socket_feedback.close()