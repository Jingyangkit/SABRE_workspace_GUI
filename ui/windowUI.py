from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow
from UserFunction import Bubbling, Gripper, State_of_the_Cooling_workspace, Experiment_start, emergency_stop, Magnetic_Field_Sweeping_parameter_Set, \
    Polarization_Relaxation_Time_parameter_Set, Polarization_Buildup_Time_parameter_Set, Pulse_Sequence_parameter_Set

from ui.window_ui import Ui_MainWindow
from utils.style import styles, changed
from utils.utils import validate_input
from experiment_motion import magnetic_field_sweeping, Polarization_Buildup_Time, Pulse_Sequence
from threading import Thread
from SABRE_Motion import RoboticArmMotion

Task = 'Q'

TaskA_Magnetic_Field_1st_value = 0
TaskA_Magnetic_Field_2nd_value = 0
TaskA_Time_Delta = 0
TaskA_Bubbling_Time = 0
TaskA_Wating_time = 0

TaskB_Low_Field = 0
TaskB_High_Field = 0
TaskB_Bubbling_Time = 0
TaskB_Wating_Time = 0

TaskC_Bubbling_1st_value = 0
TaskC_Bubbling_Time_2nd_value = 0
TaskC_Time_Delta = 0
TaskC_Magnetic_Field = 0
TaskC_Wating_Time = 0

TaskD_Low_Field = 0
TaskD_LSweeping_Time_1st_value = 0
TaskD_LSweeping_Time_2nd_value  = 0
TaskD_LSweeping_Time_Delta = 0
TaskD_High_Field  = 0
TaskD_HSweeping_Time_1st_value  = 0
TaskD_HSweeping_Time_2nd_value  = 0
TaskD_HSweeping_Time_Delta = 0
TaskD_Bubbling_Time = 0

def thread_magnetic_field_sweeping():

    expertimentA = Thread(target=magnetic_field_sweeping,
                         args=(1, TaskA_Magnetic_Field_1st_value, TaskA_Magnetic_Field_2nd_value, TaskA_Time_Delta, TaskA_Bubbling_Time, TaskA_Wating_time))
    expertimentA.setDaemon(True)
    expertimentA.start()

def thread_Polarization_Buildup_Time():

    expertimentC = Thread(target=Polarization_Buildup_Time,
                         args=(1, TaskC_Bubbling_1st_value, TaskC_Bubbling_Time_2nd_value, TaskC_Time_Delta,TaskC_Magnetic_Field, TaskC_Wating_Time))

    expertimentC.setDaemon(True)
    expertimentC.start()

# def thread_Polarization_Relaxation_Time():
#
#     expertimentB = Thread(target=Polarization_Relaxation_Time,
#                          args=(1, TaskB_Low_Field,TaskB_High_Field,TaskB_Bubbling_Time,1,TaskB_Wating_Time))
#     expertimentB.setDaemon(True)
#     expertimentB.start()

def thread_Pulse_Sequence():

    expertimentD = Thread(target=Pulse_Sequence,
                         args=(1, TaskD_Low_Field,TaskD_LSweeping_Time_1st_value,TaskD_LSweeping_Time_2nd_value,TaskD_LSweeping_Time_Delta,TaskD_High_Field,TaskD_HSweeping_Time_1st_value,TaskD_HSweeping_Time_2nd_value,TaskD_HSweeping_Time_Delta,TaskD_Bubbling_Time))
    expertimentD.setDaemon(True)
    expertimentD.start()

class MainUI(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None, *args, **kwargs):
        super(MainUI, self).__init__(parent, *args, **kwargs)
        self.setupUi(self)
        self.setWindowTitle("SABRE Workspace")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.state_worspsce = 0
        self.cycle = "nothing"
        self.cycle_all = ["TFS", "TBT", "SBT", "TA", "nothing"]
        self.styles = styles
        self.setup_ui()

    def setup_ui(self):
        """
        :return:
        """
        self.deafult_off()
        self.validate_input()
        self.pushButton_heat_on.clicked.connect(self.Bubbling_open)
        self.pushButton_heat_off.clicked.connect(self.Bubbling_close)
        self.pushButton_cool_on.clicked.connect(self.Gripper_open)
        self.pushButton_cool_off.clicked.connect(self.Gripper_close)
        self.pushButton_state.clicked.connect(self.state_codling_worspsce)
        self.pushButton_set_all.clicked.connect(self.set_all_parameters)
        self.pushButton_exp_start.clicked.connect(self.experiment_start)
        self.pushButton_exp_stop.clicked.connect(self.experiment_stop)
        self.radioButton_temp_field.clicked.connect(self.checked_model_changed)
        self.radioButton_temp_balance.clicked.connect(self.checked_model_changed)
        self.radioButton_solidifi_buildup.clicked.connect(self.checked_model_changed)
        self.radioButton_temp_alternation.clicked.connect(self.checked_model_changed)

    def validate_input(self):
        self.set_validate_input(self.lineEdit_temp_cycle_left)
        self.set_validate_input(self.lineEdit_temp_cycle_center)
        self.set_validate_input(self.lineEdit_temp_cycle_right)
        self.set_validate_input(self.lineEdit_field_heating_time)
        self.set_validate_input(self.lineEdit_field_wating_time)

        self.set_validate_input(self.lineEdit_balance_low_temp)
        self.set_validate_input(self.lineEdit_balance_high_temp)
        self.set_validate_input(self.lineEdit_balance_heating_time)
        self.set_validate_input(self.lineEdit_balance_wating_time)

        self.set_validate_input(self.lineEdit_buildup_time_left)
        self.set_validate_input(self.lineEdit_buildup_time_center)
        self.set_validate_input(self.lineEdit_buildup_time_right)
        self.set_validate_input(self.lineEdit_buildup_temp)
        self.set_validate_input(self.lineEdit_buildup_wating_time)

        self.set_validate_input(self.lineEdit_alter_low_temp)
        self.set_validate_input(self.lineEdit_alter_left)
        self.set_validate_input(self.lineEdit_alter_center)
        self.set_validate_input(self.lineEdit_alter_right)
        self.set_validate_input(self.lineEdit_alter_temp)
        self.set_validate_input(self.lineEdit_alter_heating_left)
        self.set_validate_input(self.lineEdit_alter_heating_center)
        self.set_validate_input(self.lineEdit_alter_heating_right)
        self.set_validate_input(self.lineEdit_alter_heating_down)

    def deafult_off(self):
        # self.label_state.setStyleSheet(styles[1])
        # self.label_heat.setStyleSheet(styles[1])
        # self.label_cool.setStyleSheet(styles[1])
        self.temp_field_changed(changed[0])
        self.temp_balance_changed(changed[0])
        self.solidifi_buildup_changed(changed[0])
        self.temp_alternation_changed(changed[0])

    def checked_model_changed(self):
        """temp_field"""
        checked = self.radioButton_temp_field.isChecked()
        checked2 = self.radioButton_temp_balance.isChecked()
        checked3 = self.radioButton_solidifi_buildup.isChecked()
        checked4 = self.radioButton_temp_alternation.isChecked()
        if checked:
            self.temp_field_changed(changed[1])
            self.cycle = "TFS"
        else:
            self.temp_field_changed(changed[0])
        if checked2:
            self.temp_balance_changed(changed[1])
            self.cycle = "TBT"
        else:
            self.temp_balance_changed(changed[0])
        if checked3:
            self.solidifi_buildup_changed(changed[1])
            self.cycle = "SBT"
        else:
            self.solidifi_buildup_changed(changed[0])
        if checked4:
            self.temp_alternation_changed(changed[1])
            self.cycle = "TA"
        else:
            self.temp_alternation_changed(changed[0])

    def temp_field_changed(self, changed):
        self.set_read_only(self.lineEdit_temp_cycle_left, changed)
        self.set_read_only(self.lineEdit_temp_cycle_center, changed)
        self.set_read_only(self.lineEdit_temp_cycle_right, changed)
        self.set_read_only(self.lineEdit_field_heating_time, changed)
        self.set_read_only(self.lineEdit_field_wating_time, changed)
        if changed:
            self.clear_text(self.lineEdit_temp_cycle_left)
            self.clear_text(self.lineEdit_temp_cycle_center)
            self.clear_text(self.lineEdit_temp_cycle_right)
            self.clear_text(self.lineEdit_field_heating_time)
            self.clear_text(self.lineEdit_field_wating_time)

    def temp_balance_changed(self, changed):
        self.set_read_only(self.lineEdit_balance_low_temp, changed)
        self.set_read_only(self.lineEdit_balance_high_temp, changed)
        self.set_read_only(self.lineEdit_balance_heating_time, changed)
        self.set_read_only(self.lineEdit_balance_wating_time, changed)
        if changed:
            self.clear_text(self.lineEdit_balance_low_temp)
            self.clear_text(self.lineEdit_balance_high_temp)
            self.clear_text(self.lineEdit_balance_heating_time)
            self.clear_text(self.lineEdit_balance_wating_time)

    def solidifi_buildup_changed(self, changed):
        self.set_read_only(self.lineEdit_buildup_time_left, changed)
        self.set_read_only(self.lineEdit_buildup_time_center, changed)
        self.set_read_only(self.lineEdit_buildup_time_right, changed)
        self.set_read_only(self.lineEdit_buildup_temp, changed)
        self.set_read_only(self.lineEdit_buildup_wating_time, changed)
        if changed:
            self.clear_text(self.lineEdit_buildup_time_left)
            self.clear_text(self.lineEdit_buildup_time_center)
            self.clear_text(self.lineEdit_buildup_time_right)
            self.clear_text(self.lineEdit_buildup_temp)
            self.clear_text(self.lineEdit_buildup_wating_time)

    def temp_alternation_changed(self, changed):
        self.set_read_only(self.lineEdit_alter_low_temp, changed)
        self.set_read_only(self.lineEdit_alter_left, changed)
        self.set_read_only(self.lineEdit_alter_center, changed)
        self.set_read_only(self.lineEdit_alter_right, changed)
        self.set_read_only(self.lineEdit_alter_temp, changed)
        self.set_read_only(self.lineEdit_alter_heating_left, changed)
        self.set_read_only(self.lineEdit_alter_heating_center, changed)
        self.set_read_only(self.lineEdit_alter_heating_right, changed)
        self.set_read_only(self.lineEdit_alter_heating_down, changed)
        if changed:
            self.clear_text(self.lineEdit_alter_low_temp)
            self.clear_text(self.lineEdit_alter_left)
            self.clear_text(self.lineEdit_alter_center)
            self.clear_text(self.lineEdit_alter_right)
            self.clear_text(self.lineEdit_alter_temp)
            self.clear_text(self.lineEdit_alter_heating_left)
            self.clear_text(self.lineEdit_alter_heating_center)
            self.clear_text(self.lineEdit_alter_heating_right)
            self.clear_text(self.lineEdit_alter_heating_down)

    def read_temp_field(self):
        temp_cycle_left = self.read_text(self.lineEdit_temp_cycle_left)
        temp_cycle_center = self.read_text(self.lineEdit_temp_cycle_center)
        temp_cycle_right = self.read_text(self.lineEdit_temp_cycle_right)
        field_heating_time = self.read_text(self.lineEdit_field_heating_time)
        field_wating_time = self.read_text(self.lineEdit_field_wating_time)
        # if temp_cycle_left != '' and temp_cycle_center != '' and temp_cycle_right != '' and field_heating_time != '' and field_wating_time != '':
        return [temp_cycle_left, temp_cycle_center, temp_cycle_right, field_heating_time, field_wating_time]

    def read_temp_balance(self):
        low_temp = self.read_text(self.lineEdit_balance_low_temp)
        high_temp = self.read_text(self.lineEdit_balance_high_temp)
        heating_time = self.read_text(self.lineEdit_balance_heating_time)
        wating_time = self.read_text(self.lineEdit_balance_wating_time)
        # if low_temp != '' and high_temp != '' and heating_time != '' and wating_time != '':
        return [low_temp, high_temp, heating_time, wating_time]

    def read_solidifi_buildup(self):
        time_left = self.read_text(self.lineEdit_buildup_time_left)
        time_center = self.read_text(self.lineEdit_buildup_time_center)
        time_right = self.read_text(self.lineEdit_buildup_time_right)
        buildup_temp = self.read_text(self.lineEdit_buildup_temp)
        wating_time = self.read_text(self.lineEdit_buildup_wating_time)
        # if time_left != '' and time_center != '' and time_right != '' and buildup_temp != '' and wating_time != '':
        return [time_left, time_center, time_right, buildup_temp, wating_time]

    def read_temp_alternation(self):
        low_temp = self.read_text(self.lineEdit_alter_low_temp)
        alter_left = self.read_text(self.lineEdit_alter_left)
        alter_center = self.read_text(self.lineEdit_alter_center)
        alter_right = self.read_text(self.lineEdit_alter_right)
        alter_temp = self.read_text(self.lineEdit_alter_temp)
        heating_left = self.read_text(self.lineEdit_alter_heating_left)
        heating_center = self.read_text(self.lineEdit_alter_heating_center)
        heating_right = self.read_text(self.lineEdit_alter_heating_right)
        heating_down = self.read_text(self.lineEdit_alter_heating_down)
        # if low_temp != '' and alter_left != '' and alter_center != '' and alter_right != '' and alter_temp != '' and heating_left != '' and heating_center != '' and heating_right != '' and heating_down != '':
        return [low_temp, alter_left, alter_center, alter_right, alter_temp, heating_left, heating_center, heating_right, heating_down]

    def set_validate_input(self, lineEdit):
        validate_input(lineEdit)

    def set_read_only(self, lineEdit, changed):
        lineEdit.setEnabled(not changed)
        lineEdit.setReadOnly(changed)

    def clear_text(self, lineEdit):
        lineEdit.setText('')

    def read_text(self, lineEdit):
        return lineEdit.text()

    def set_all_parameters(self):
        if self.cycle == self.cycle_all[0]:
            temp_parameters = self.read_temp_field()
            Magnetic_Field_Sweeping_parameter_Set(
                Magnetic_Field_1st_value=temp_parameters[0],
                Magnetic_Field_2nd_value=temp_parameters[1],
                Time_Delta=temp_parameters[2],
                Bubbling_Time=temp_parameters[3],
                Wating_time=temp_parameters[4]
            )
            global TaskA_Magnetic_Field_1st_value
            TaskA_Magnetic_Field_1st_value = temp_parameters[0]
            global TaskA_Magnetic_Field_2nd_value
            TaskA_Magnetic_Field_2nd_value = temp_parameters[1]
            global TaskA_Time_Delta
            TaskA_Time_Delta = temp_parameters[2]
            global TaskA_Bubbling_Time
            TaskA_Bubbling_Time = temp_parameters[3]
            global TaskA_Wating_time
            TaskA_Wating_time = temp_parameters[4]
            global Task
            Task = 'A'

        if self.cycle == self.cycle_all[1]:
            temp_parameters = self.read_temp_balance()
            Polarization_Relaxation_Time_parameter_Set(
                Low_Field=temp_parameters[0],
                High_Field=temp_parameters[1],
                Bubbling_Time=temp_parameters[2],
                Wating_Time=temp_parameters[3]
            )
            global TaskB_Low_Field
            TaskB_Low_Field = temp_parameters[0]
            global TaskB_High_Field
            TaskB_High_Field = temp_parameters[1]
            global TaskB_Bubbling_Time
            TaskB_Bubbling_Time = temp_parameters[2]
            global TaskB_Wating_Time
            TaskB_Wating_Time = temp_parameters[3]
            global Task
            Task = 'B'

        if self.cycle == self.cycle_all[2]:
            temp_parameters = self.read_solidifi_buildup()
            Polarization_Buildup_Time_parameter_Set(
                Bubbling_1st_value=temp_parameters[0],
                Bubbling_Time_2nd_value=temp_parameters[1],
                Time_Delta=temp_parameters[2],
                Magnetic_Field=temp_parameters[3],
                Wating_Time=temp_parameters[4]
            )
            global TaskC_Bubbling_1st_value
            TaskC_Bubbling_1st_value = temp_parameters[0]
            global TaskC_Bubbling_Time_2nd_value
            TaskC_Bubbling_Time_2nd_value = temp_parameters[1]
            global TaskC_Time_Delta
            TaskC_Time_Delta = temp_parameters[2]
            global TaskC_Magnetic_Field
            TaskC_Magnetic_Field = temp_parameters[3]
            global TaskC_Wating_Time
            TaskC_Wating_Time = temp_parameters[4]
            global Task
            Task = 'C'

        if self.cycle == self.cycle_all[3]:
            temp_parameters = self.read_temp_alternation()
            Pulse_Sequence_parameter_Set(
                Low_Field=temp_parameters[0],
                LSweeping_Time_1st_value=temp_parameters[1],
                LSweeping_Time_2nd_value=temp_parameters[2],
                LSweeping_Time_Delta=temp_parameters[3],
                High_Field=temp_parameters[4],
                HSweeping_Time_1st_value=temp_parameters[5],
                HSweeping_Time_2nd_value=temp_parameters[6],
                HSweeping_Time_Delta=temp_parameters[7],
                Bubbling_Time=temp_parameters[8]
            )
            global TaskD_Low_Field
            TaskD_Low_Field = temp_parameters[0]
            global TaskD_LSweeping_Time_1st_value
            TaskD_LSweeping_Time_1st_value = temp_parameters[1]
            global TaskD_LSweeping_Time_2nd_value
            TaskD_LSweeping_Time_2nd_value = temp_parameters[2]
            global TaskD_LSweeping_Time_Delta
            TaskD_LSweeping_Time_Delta = temp_parameters[3]
            global TaskD_High_Field
            TaskD_High_Field = temp_parameters[4]
            global TaskD_HSweeping_Time_1st_value
            TaskD_HSweeping_Time_1st_value = temp_parameters[5]
            global TaskD_HSweeping_Time_2nd_value
            TaskD_HSweeping_Time_2nd_value = temp_parameters[6]
            global TaskD_HSweeping_Time_Delta
            TaskD_HSweeping_Time_Delta = temp_parameters[7]
            global TaskD_Bubbling_Time
            TaskD_Bubbling_Time = temp_parameters[8]
            global Task
            Task = 'D'

        if self.cycle == self.cycle_all[4]:
            print("The experiment configuration was not selected correctly")

    def state_codling_worspsce(self):
        if self.state_worspsce == 0:
            self.state_worspsce = 1
            self.label_state.setStyleSheet(styles[0])
            self.state_cooling_workspace(0)
        else:
            self.state_worspsce = 0
            self.label_state.setStyleSheet(styles[1])
            self.state_cooling_workspace(1)

    def Bubbling_open(self):
        self.label_heat.setStyleSheet(styles[0])
        self.Bubbling_changed(0)
        
    def Bubbling_close(self):
        self.label_heat.setStyleSheet(styles[1])
        self.Bubbling_changed(1)

    def Gripper_open(self):
        self.label_cool.setStyleSheet(styles[0])
        self.Gripper_changed(0)

    def Gripper_close(self):
        self.label_cool.setStyleSheet(styles[1])
        self.Gripper_changed(1)

    def Bubbling_changed(self, state):
        Bubbling(state)

    def Gripper_changed(self, state):
        Gripper(state)

    def state_cooling_workspace(self, state):
        State_of_the_Cooling_workspace(state)

    def experiment_start(self):
        Experiment_start()
        if Task == 'A':
            thread_magnetic_field_sweeping()
        # elif Task == 'B':
        #     thread_Polarization_Relaxation_Time()
        elif Task == 'C':
            thread_Polarization_Buildup_Time()
        elif Task == 'D':
            thread_Pulse_Sequence()

    def experiment_stop(self):
        emergency_stop()
        RoboticArmMotion.disable(1)




