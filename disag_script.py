# Script disaggregates REDD building data based on input arguments
# Author: Michael Milicevich
# Date: 16/02/2015

# command linen test variables to use:
# python disag_script.py 1 "fridge" "2011-05-01" "2011-05-02"

# import modules/dependencies
from __future__ import print_function, division
import sys
from dateutil import parser
import pandas as pd
from key_map import *
from nilmtk import HDFDataStore, DataSet, TimeFrame, MeterGroup
from nilmtk.disaggregate import CombinatorialOptimisation
import warnings

#supress warnings to users console
warnings.filterwarnings("ignore")

# verify length of args, should be 5 corrsponding to:
#[0]: script name: disag_script.py
#[1]: REDD Building : redd_building
#[2]: Appliance to disaggregate: disag_appliances
#[3]: Beginning time/date: t1
#[4]: Ending time/date: t2

# Verify that the correct amount of input arguments have been entered
if len(sys.argv) != 5:
	sys.exit("Error: Incorrect amount of input arguments given. Script terminated.")

#load arguments into script
redd_building = int(sys.argv[1])
disag_appliance = sys.argv[2]
t1 = sys.argv[3] + " 00:00:00+00:00"
t2 = sys.argv[4] + " 00:00:00+00:00"

# to add:
#			1) load REDD data from database (SQL interface)*
#
#			*Cannot be implemented until database is setup in environment

# Verify input appliance exists in building
km = Key_Map(redd_building)

# verify a real appliance has been entered
if km.is_in_map(disag_appliance) == False:
	sys.exit("An incorrect appliance name has been entered. Please ensure the entered name is exactly correct.")

redd_data = DataSet("C:/NILM/Data/REDD/redd.h5")

# load mains of the building
building_mains = redd_data.buildings[redd_building].elec.mains()

#train disaggregation set
co = CombinatorialOptimisation()
training_set = redd_data.buildings[redd_building].elec
co.train(training_set)

#set output datastore
outputData = HDFDataStore("C:/NILM/Data/Output/output.h5",'w')

#disaggregate
co.disaggregate(building_mains,outputData)

# to add:
#			1) get the meter instance # of the appliance selected
#			2) export the meter instance series of the output datastore to database using SQL, within t1-t2 parameters*
#
#			*Cannot be implemented until database is setup in environment


#Close open datastores
redd_data.store.close()
outputData.store.close()


