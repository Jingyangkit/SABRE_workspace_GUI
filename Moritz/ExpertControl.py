# Author: Craig Eccles (Magritek), 23.05.2021
# Changed by: Moritz Becker (IMT), 26.05.2021
# Demonstrates how to start SpinsolveExpert and then run different experiments or commands
# via Windows message passing. Tested using Python 3.8 with Anaconda
import win32con, win32api, win32gui, win32ui
import ctypes, ctypes.wintypes
import subprocess,time
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

# Prospa executable and SpinsolveExpert macro location - change as required
PROSPA_PATH = 'C:\\Users\\Magritek\\Applications\\SpinsolveExpert 1.41.06\\prospa.exe'
MACRO_PATH = 'C:\\Users\\Magritek\\Applications\\SpinsolveExpert 1.41.06\\Macros\\Spinsolve-Expert\\SpinsolveExpertInterface.pex'
EXCHANGE_FILE = 'C:\\Users\\Magritek\\Documents\\Moritz\\ExchangeVariables.par'

# Defines data structure used to pass information between Python and Prospa
class COPYDATASTRUCT(ctypes.Structure):
    _fields_ = [
        ('dwData', ctypes.wintypes.LPARAM),   # User define parameter
        ('cbData', ctypes.wintypes.DWORD),    # Size of string
        ('lpData', ctypes.c_char_p)           # String containing data
    ]
PCOPYDATASTRUCT = ctypes.POINTER(COPYDATASTRUCT)

# A  class allowing communications between Python and Prospa
class Comms:

   # Defines an invisible win32 window to receive and send messages
    def __init__(self, dstWin):
        message_map = {
            win32con.WM_COPYDATA: self.OnUser
        }
        wc = win32gui.WNDCLASS()
        wc.lpfnWndProc = message_map
        wc.lpszClassName = 'MyWindowClass'
        hinst = wc.hInstance = win32api.GetModuleHandle(None)
        classAtom = win32gui.RegisterClass(wc)
        self.hwnd = win32gui.CreateWindow (
            classAtom,
            "win32gui test",
            0,
            0,
            0,
            win32con.CW_USEDEFAULT,
            win32con.CW_USEDEFAULT,
            0,
            0,
            hinst,
            None
        )
        self.returnMessage = 'init'
        self.prospaWin = dstWin

   # Detect a message coming back from Prospa
    def OnUser(self, hwnd, msg, wparam, lparam):
        pCDS = ctypes.cast(lparam, PCOPYDATASTRUCT)
       # print (pCDS.contents.dwData)
       # print (pCDS.contents.cbData)
        self.returnMessage = ctypes.string_at(pCDS.contents.lpData)
        return 1

   # Run a macro or command in Prospa by sending the text as a message
    def RunProspaMacro(self, macro_str):
        pywnd = win32ui.CreateWindowFromHandle(self.prospaWin)
        cds = COPYDATASTRUCT()
        cds.dwData = 1
        cds.cbData = ctypes.sizeof(ctypes.create_string_buffer(macro_str))
        cds.lpData = ctypes.c_char_p(macro_str)
        lParam = PCOPYDATASTRUCT.from_address(ctypes.addressof(cds))
        pywnd.SendMessage(win32con.WM_COPYDATA, self.hwnd, lParam)
        
        
def yes_or_no(question):
    while "the answer is invalid":
        reply = str(input(question+' (y/n): ')).lower().strip()
        if reply[:1] == 'y':
            return True
        if reply[:1] == 'n':
            return False

# Open SpinsolveExpert - need V1.41.06 or later
# Start Prospa ('false' for hidden 'true' for visible). Give it time to load
subprocess.Popen([PROSPA_PATH, MACRO_PATH,'"false"'])
time.sleep(3)
# Find the Prospa window
prospaWin = win32gui.FindWindowEx(0, 0, 'PROSPAWIN', None)
# Make a comms class object to communicate with prospa
com = Comms(prospaWin)

print("Experiments started\n")

# Perform dummy Proton experiment in the start. (Otherwise paths do not match.)
# Load the first experiment in to the interface
com.RunProspaMacro(b'Proton(["nrScans = 1"])')
# Change the current file comment
com.RunProspaMacro(b'gView->sampleNameCtrl->text("CuSO4")')
# Run the first experiment
com.RunProspaMacro(b'gExpt->runExperiment()')

count = 1 # my user variable
# Write exchange variables to file
# Make sure to match prospa writing
with open(EXCHANGE_FILE, "w") as f:
    f.write("counter = {}".format(count))

# execute ExternalRun macro. This will fetch "exchange" variables
com.RunProspaMacro("ExternalRun".encode())
# Get data location
output_dir = com.returnMessage

print("Experiment data in ", output_dir)

print("Experiments finished\n")

# Remove dummy scan data 
if yes_or_no("Remove dummy data? "):
    files = glob.glob(output_dir.decode()+"\\*.*")
    for f in files:
        os.remove(f)

# Close Expert after 2 seconds
com.RunProspaMacro(b'showwindow(0)')
time.sleep(2)
com.RunProspaMacro(b'exit(0)')

# Keep plots
plt.show()

# Some other commands

# com.RunProspaMacro(b'ReactionMonitoring(["nrSteps = 5","nrScans = 2","ppmRange = [-2,12]", "repTime=2000"])')
# com.RunProspaMacro(b'gExpt->runExperiment()')
# print(com.returnMessage)


#RunProspaMacro(win,b'QuickShim(["peakPositionPPM = 1","startMethod=\\"last\\"","shimMethod=\\"order12\\""])')
#RunProspaMacro(win,b'LockAndCalibrate(["refPeakPPM = 1"])')
