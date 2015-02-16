# Script disaggregates REDD building data based on input arguments
# Author: Michael Milicevich
# Date: 16/02/2015

# sys used to parse inoput arguments
import sys

# verify length of args, should be 5 corrsponding to:
#[0]: script name: disag_script.py
#[1]: REDD Building : arg_redd_building
#[2]: Appliance to disaggregate: arg_appliances
#[3]: Beginning time/date: arg_t1
#[4]: Ending time/date: arg_t2

# Verify that the correct amount of input arguments have been entered
if len(sys.argv) != 5:
	sys.exit("Error: Incorrect amount of input arguments given. Script terminated.")


