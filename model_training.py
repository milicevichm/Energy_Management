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

#declare key map object set to building 1
km = Key_Map(1)

#declare REDD datastore
redd_data = DataSet("C:/NILM/Data/redd_data.h5")

'''
Unable to get select_top_k() function working so must manually declare each appliance in its own metergroup in
order to get combinations for testing different training groups.

'''
fridge = redd_data.buildings[1].elec.select_using_appliances(type='fridge')
dish_washer = redd_data.buildings[1].elec.select_using_appliances(type='dish washer')
sockets = redd_data.buildings[1].elec.select_using_appliances(type='sockets')
lights =  redd_data.buildings[1].elec.select_using_appliances(type='lights')
unknowns = redd_data.buildings[1].elec.select_using_appliances(type='unknown')
microwave = redd_data.buildings[1].elec.select_using_appliances(type='microwave')
electric_space_heater = redd_data.buildings[1].elec.select_using_appliances(type='electric space heater')
electric_stove = redd_data.buildings[1].elec.select_using_appliances(type='electric stove')
washer_dryer = redd_data.buildings[1].elec.select_using_appliances(type='washer dryer')
electric_oven = redd_data.buildings[1].elec.select_using_appliances(type='electric oven')

'''
Display total energy per submeter group (so we can rank by total meter energy)

'''
print("fridge: "+str(fridge.total_energy()))
print("dishwasher: "+str(dish_washer.total_energy()))
print("sockets: "+str(sockets.total_energy()))
print("lights: "+str(lights.total_energy()))
print("unknowns:"+str(unknowns.total_energy()))



training_set1 = fridge.union(fridge)



#declare disaggregation algorithm
co = CombinatorialOptimisation()

#train algorithm model using training_set
#co.train(training_set1)



#Close dataset dataStore
redd_data.store.close()