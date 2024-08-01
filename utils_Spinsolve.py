# Utils to reduce load in python script

# Author: Craig Eccles (Magritek), 23.05.2021
# Changed by: Moritz Becker (IMT), 26.05.2021
# Demonstrates how to start SpinsolveExpert and then run different experiments or commands
# via Windows message passing. Tested using Python 3.8 with Anaconda
import os
import glob
import win32con, win32api, win32gui, win32ui
import ctypes, ctypes.wintypes
import subprocess,time
import numpy as np
import nmrglue as ng
import matplotlib.pyplot as plt
from scipy import signal
import time

# Prospa executable and SpinsolveExpert macro location - change as required
#PROSPA_PATH = 'C:\\Users\\Magritek\\Applications\\SpinsolveExpert 1.41.06\\prospa.exe'
PROSPA_PATH = 'C:\\Users\\Magritek\\Applications\\SpinsolveExpert\\prospa.exe'
#MACRO_PATH = 'C:\\Users\\Magritek\\Applications\\SpinsolveExpert 1.41.06\\Macros\\Spinsolve-Expert\\SpinsolveExpertInterface.pex'
MACRO_PATH = 'C:\\Users\\Magritek\\Applications\\SpinsolveExpert\\Macros\\Spinsolve-Expert\\SpinsolveExpertInterface.pex'
#EXCHANGE_FILE = 'C:\\Users\\Magritek\\Documents\\Moritz\\ExchangeVariables.par'
EXCHANGE_FILE = 'C:\\Users\\Magritek\\Documents\\PythonScripts\\ExchangeVariables.par'

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
            
            
def init(verbose=False, gui=False):
    # Open SpinsolveExpert - need V1.41.06 or later
    # Start Prospa ('false' for hidden 'true' for visible). Give it time to load
    if gui == False: subprocess.Popen([PROSPA_PATH, MACRO_PATH,'"false"'])
    if gui == True: subprocess.Popen([PROSPA_PATH, MACRO_PATH,'"true"'])
    time.sleep(3)
    # Find the Prospa window
    prospaWin = win32gui.FindWindowEx(0, 0, 'MAIN_PROSPAWIN', None)
    # Make a comms class object to communicate with prospa
    com = Comms(prospaWin)
    
    if verbose: print("Dummy experiments started\n")

    # Perform dummy Proton experiment in the start. (Otherwise paths do not match.)
    # Load the first experiment in to the interface
    com.RunProspaMacro(b'Proton(["nrScans = 1"])')
    # Change the current file comment
    com.RunProspaMacro(b'gView->sampleNameCtrl->text("PythonShim")')
    # Run the first experiment
    com.RunProspaMacro(b'gExpt->runExperiment()')
    
    return com

# height: 0.5 or 0.9945 (1-x)
def get_linewidth_Hz(spectrum, sampling_points=32768, bandwidth = 20000, height=0.5):
    peak_index = signal.find_peaks(spectrum, height = spectrum.max()*0.9, distance=5000)[0]
    [width, height_of_evaluation,_,_] = signal.peak_widths(spectrum, peak_index, rel_height=height)
    return bandwidth/sampling_points*width
    
def get_RMS(fid):
    return np.sqrt(np.mean(np.square(abs(fid))))
    
       
        
# Modification: Allows all shims to be changed
def setShimsAndRunV3(com, count, shim_values_in, return_fid = False, return_shimvalues=False, verbose=False,
                     lowRes = False):
    nr_shims = 15
    shim_values = np.zeros(nr_shims, dtype=int)
    shim_values[:len(shim_values_in)]=shim_values_in
    with open(EXCHANGE_FILE, "w") as f:
        f.write("counter = {}\n".format(count))
        for i in range(nr_shims):
            f.write("d{} = {}\n".format(i+1, shim_values[i]))
        
    # execute ExternalRun macro. This will fetch "exchange" variables
    if not lowRes: 
        com.RunProspaMacro("Moritz_ExternalCustomShimV3".encode())
    else:
        com.RunProspaMacro("Moritz_ExternalCustomShimV3LowRes".encode())
    
    # Get data location
    output_dir = com.returnMessage
    if verbose: print('Output dir (unformated): ',output_dir)
    
    dic, fid = ng.spinsolve.read(output_dir.decode() + '/{}/'.format(count))
    # more uniform listing
    udic = ng.spinsolve.guess_udic(dic, fid)
    # fft and phase correction
    spectrum = ng.proc_base.fft(fid)
    spectrum = ng.proc_base.ps(spectrum, p0=float(dic['proc']['p0Phase']) ,p1=float(dic['proc']['p1Phase']))
    spectrum = ng.proc_base.di(spectrum)
    
    xaxis = np.arange(-udic[0]['sw']/2,udic[0]['sw']/2, udic[0]['sw']/udic[0]['size'])
       
    if return_shimvalues: 
        if verbose: print('Returning shim values')
        shims_dic = np.array([ { line.split()[0] : int(line.split()[2]) for line in open(output_dir.decode() + '/{}/shims.par'.format(count))}])[0]   
        shims = [ shims_dic['{}'.format(d)] for d in shims_dic]
        if return_fid: return xaxis, spectrum, fid, shims
        return xaxis, spectrum, shims
    else:
        if return_fid: return xaxis, spectrum, fid
        return xaxis, spectrum
    
    
def readExchangeFile():
    output = []
    with open(EXCHANGE_FILE, "w") as f:
        for content in f: 
            output.append(f.read())           
    return output
 
    
def shutdown(com, verbose=False):
    
    com.RunProspaMacro(b'showwindow(0)')
    time.sleep(2)
    com.RunProspaMacro(b'exit(0)')
    
    if verbose: print('Shutdown successfull.')
