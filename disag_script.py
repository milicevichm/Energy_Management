# Script disaggregates REDD building data based on input arguments
# Author: Michael Milicevich
# Date: 16/02/2015

# sys used to parse inoput arguments
import sys
from dateutil import parser

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
redd_building = sys.argv[1]
disag_appliances = sys.argv[2]
t1 = parser.parse(sys.argv[3])
t2 = parser.parse(sys.argv[4])

