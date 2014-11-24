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

'''
for i in redd_data.buildings:
	elec = redd_data.buildings[i].elec
	do_rate = elec.dropout_rate()
	print("Building "+str(i)+" Total Dropout Rate: "+str(do_rate))


print("\n")


for i in redd_data.buildings:
	if i != 3:
		elec = redd_data.buildings[i].elec
		gs = elec.good_sections(full_results = True)
		do_rate = elec.dropout_rate(sections = gs.combined())
		print("Building "+str(i)+" Adjusted Dropout Rate: "+str(do_rate))
	else:
		print("Building 3 Adjusted Dropout Rate: N/A")


print("\n")


for i in redd_data.buildings:
	elec = redd_data.buildings[i].elec
	percent_sm = elec.proportion_of_energy_submetered()
	print("Building "+str(i)+" Percent of Energy Submetered: "+str(percent_sm))

'''

redd_data.buildings[5].elec.mains().plot()
plt.title("Building 5 Mains Power Energy")
plt.ylabel('Real Power [W]')
plt.xlabel('Time')
plt.show()

redd_data.buildings[1].elec.mains().plot()
plt.title("Building 1 Mains Power Energy")
plt.ylabel('Real Power [W]')
plt.xlabel('Time')
plt.show()

redd_data.store.close()