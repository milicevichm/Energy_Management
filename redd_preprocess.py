#Author: Michael Milicevich
# Script analyses dropout rate between all 6 REDD buildings
#

from __future__ import print_function, division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nilmtk import HDFDataStore, DataSet
from nilmtk.metergroup import *
import warnings

#supress warnings to users console
warnings.filterwarnings("ignore")

print("Loading Dataset...")
#declare REDD datastore
redd_data = DataSet("C:/NILM/Data/redd_data.h5")
#redd_data.store.window = TimeFrame(start = "2011-05-01 00:00:00+00:00", end = "2011-05-01 12:00:00+00:00")
print("Dataset Loaded!\n")

for i in redd_data.buildings:
	elec = redd_data.buildings[i].elec
	do_rate = elec.dropout_rate()
	print("Building "+str(i)+" Total Dropout Rate: "+str(do_rate))


elec1 = redd_data.buildings[1].elec
gs1 = elec1.good_sections(full_results = True)
do_rate1 = elec1.dropout_rate(sections = gs1.combined())
print ("\n\nBuilding 1 Adjusted Dropout Rate: " + str(do_rate1))

elec2 = redd_data.buildings[2].elec
gs2 = elec2.good_sections(full_results = True)
do_rate2 = elec2.dropout_rate(sections = gs2.combined())
print ("Building 2 Adjusted Dropout Rate: " + str(do_rate2))

elec3 = redd_data.buildings[3].elec
gs3 = elec3.good_sections(full_results = True)
do_rate3 = elec3.dropout_rate(sections = gs3.combined())
print ("Building 3 Adjusted Dropout Rate: " + str(do_rate3))

elec4 = redd_data.buildings[4].elec
gs4 = elec4.good_sections(full_results = True)
do_rate4 = elec4.dropout_rate(sections = gs4.combined())
print ("Building 4 Adjusted Dropout Rate: " + str(do_rate4))

elec5 = redd_data.buildings[5].elec
gs5 = elec5.good_sections(full_results = True)
do_rate5 = elec5.dropout_rate(sections = gs5.combined())
print ("Building 5 Adjusted Dropout Rate: " + str(do_rate5))

elec6 = redd_data.buildings[6].elec
gs6 = elec6.good_sections(full_results = True)
do_rate6 = elec6.dropout_rate(sections = gs6.combined())
print ("Building 6 Adjusted Dropout Rate: " + str(do_rate6))


redd_data.store.close()