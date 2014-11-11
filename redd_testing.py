#aAuthor: Michael Milicevich
#Program is a test script for REDD energy data disaggregation.

from redd_data import *
from key_map import *

'''

To change the program alter the variables below in the box:

redd_fp: 					The complete filepath to the redd_data.h5 file (included in the project github under the folder 'Data').
output_fp: 					The complete filepath to the output file that is created during disaggregation.
training_building_inst:		The building instance (1-6) that you wish to use to train the disaggregation algorithm.
disag_building_inst:		The building instance (1-6) that you wish to dissaggregate.
appliance_name:				The name of the appliance that you wish to plot for disaggregation. (Cannot be mains1 or mains2)
								*the name of all applicable appliances can be found by using the list_applianceS() method for key_map object. 
								*each building will have a different set of appliances, names can vary.
t1:							The starting timestamp for plotting data
t2:							The ending timestamp for plotting data

'''

#Hardcoded Variables --------------------------------------------
redd_fp = "C:/NILM/Data/redd_data.h5"				#
output_fp = "C:/NILM/Data/output_data.h5"			#
training_building_inst = 1;										#
disag_building_inst = 1;										#
appliance_name = "fridge"									#
t1 = "2011-05-1 1:00"											#
t2 = "2011-05-01 12:00"											#
#----------------------------------------------------------------

#initialize Redd Data object
redd_data = REDD_Data(redd_fp,output_fp)

#train Disaggregation Algorithm using all appliances on chosen building
redd_data.train_disag_model(training_building_inst)

#disaggregate building data using trained model
redd_data.disaggregate(disag_building_inst)

#plot mains data
#redd_data.plot_mains(disag_building_inst,t1,t2,1)

#print redd metadata
redd_data.plot_redd_mains_data(1,t1,t2)

#plot all appliances per building
#redd_data.building_plot_all(disag_building_inst,t1,t2)

#plot disaggregted appliance data
#redd_data.plot_disag_apl(disag_building_inst,appliance_name,t1,t2)

#show both plots
redd_data.show_plots()

#close datastores
redd_data.close()