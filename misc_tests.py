#Author: Michael Milicevich
# Script defines and trains the machine learning module
# Used for finding optimal training set (for REDD)

from __future__ import print_function, division
from nilmtk import HDFDataStore, DataSet, TimeFrame
from nilmtk.disaggregate import CombinatorialOptimisation
import pandas as pd
import matplotlib.pyplot as plt
from nilmtk.plots import plot_series
from key_map import *
from nilmtk.metrics import *
import warnings

#supress warnings to users console
warnings.filterwarnings("ignore")

#create numbered dictonary of meter# to appliance name


t1 = "2011-05-1 1:00"
t2 = "2011-05-01 12:00"	

#declate key_map object
km = Key_Map(1)

#load original redd data
print("Loading REDD and Disaggregated Datasets...")
redd_data = DataSet("C:/NILM/Data/redd_data.h5")
out1 = DataSet("C:/NILM/Data/Model_Train/output1.h5")
out2 = DataSet("C:/NILM/Data/Model_Train/output2.h5")
out3 = DataSet("C:/NILM/Data/Model_Train/output3.h5")
out4 = DataSet("C:/NILM/Data/Model_Train/output4.h5")
out5 = DataSet("C:/NILM/Data/Model_Train/output5.h5")
out6 = DataSet("C:/NILM/Data/Model_Train/output6.h5")
out7 = DataSet("C:/NILM/Data/Model_Train/output7.h5")
out8 = DataSet("C:/NILM/Data/Model_Train/output8.h5")
out9 = DataSet("C:/NILM/Data/Model_Train/output9.h5")
print("DataSets loaded!")

appliance = 'fridge'

#plot appliance's actual fingerprint vs predicted fingerprint
plot_series(redd_data.store.store.get(km.get_key(appliance))[t1:t2])
plot_series(out5.store.store.get(km.get_key(appliance))[t1:t2])
plt.legend(['Actual Measurement','Estimated Measurement'])
plt.title("Measured "+appliance.capitalize()+" Energy vs Estimated "+appliance.capitalize()+" Energy")
plt.show()

