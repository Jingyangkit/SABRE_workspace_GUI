import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import pickle

 


MYPATH = str(Path(__file__).parents[0])
sys.path.append(MYPATH)
DATAPATH = MYPATH + '\data\CeDR\dataStats'
print(MYPATH)
 

from utils_IO import get_dataset
DATAFILE = '/230525-152857_dataset_XYZZ2_Ref_range50_MONITOR'

#data_all, labels, norm_factor, max_data, xaxis = get_dataset(DATAPATH, ROI_min= 16030)

def preLoadData(DATAFILE):
    data_all, labels, norm_factor, max_data, xaxis = get_dataset(DATAPATH + DATAFILE, ROI_min= 16030, nr_shims = 4)

    with open(DATAPATH + "/preLoaded" + DATAFILE + '.pickle', "wb") as handle:
        pickle.dump([data_all, labels], handle)


preLoadData(DATAFILE)




