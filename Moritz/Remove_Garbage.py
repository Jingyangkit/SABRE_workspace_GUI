# Place this script in root of dataset and run to delete all .pt1 files

import os
import glob

cwd = os.getcwd()

def yes_or_no(question):
    while "the answer is invalid":
        reply = str(input(question+' (y/n): ')).lower().strip()
        if reply[:1] == 'y':
            return True
        if reply[:1] == 'n':
            return False
            

files_fid = glob.glob(cwd+"/*/fid.pt1")
files_spec = glob.glob(cwd+"/*/spectrum.pt1")
files_2 = glob.glob(cwd+"/*/*.par.bak")
files_3 = glob.glob(cwd+"/*/*.script")


if yes_or_no("Remove Garbage? "):

    for f in files_fid:
        os.remove(f)
    for f in files_spec:
        os.remove(f)
    for f in files_2:
        os.remove(f)
    for f in files_3:
        os.remove(f)
        
        
    print("Done")