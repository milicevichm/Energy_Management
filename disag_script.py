# Script disaggregates REDD building data based on input arguments
# Author: Michael Milicevich
# Date: 16/02/2015

# import modules/dependencies
from __future__ import print_function, division
import sys
from dateutil import parser
import pandas as pd
from key_map import *
from nilmtk import *
import warnings


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
disag_appliances = sys.argv[2]
t1 = parser.parse(sys.argv[3])
t2 = parser.parse(sys.argv[4])

# to add:
#			1) load REDD data from database (SQL interface)*
#			2) load training model based on building number
#			3) disaggregate the data
#			4) output data into database (SQL interface)*
#
#			*Cannot be implemented until database is setup in environment

#Verify input appliance exists in building
km = Key_Map(redd_building)

# verify a real appliance has been entered
if km.is_in_map(disag_appliances) == False:
	sys.exit("An incorrect appliance name has been entered. Please ensure the entered name is exactly correct.")


print("Loading REDD Dataset...")
#redd_data = DataSet("C:/NILM/Data/REDD/redd.h5")

