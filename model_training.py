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
import warnings

#supress warnings to users console
warnings.filterwarnings("ignore")

#declare key map object set to building 1
km = Key_Map(1)

print("Loading Dataset...")
#declare REDD datastore
redd_data = DataSet("C:/NILM/Data/redd_data.h5")
#redd_data.store.window = TimeFrame(start = "2011-05-01 00:00:00+00:00", end = "2011-05-01 12:00:00+00:00")
print("Dataset Loaded!")

'''
Unable to get select_top_k() function working so must manually declare each appliance in its own metergroup in
order to get combinations for testing different training groups.

'''

fridge = redd_data.buildings[1].elec.select_using_appliances(type='fridge')
dish_washer = redd_data.buildings[1].elec.select_using_appliances(type='dish washer')
sockets = redd_data.buildings[1].elec.select_using_appliances(type='sockets')
lights =  redd_data.buildings[1].elec.select_using_appliances(type='light')
unknowns = redd_data.buildings[1].elec.select_using_appliances(type='unknown')
microwave = redd_data.buildings[1].elec.select_using_appliances(type='microwave')
electric_space_heater = redd_data.buildings[1].elec.select_using_appliances(type='electric space heater')
electric_stove = redd_data.buildings[1].elec.select_using_appliances(type='electric stove')
washer_dryer = redd_data.buildings[1].elec.select_using_appliances(type='washer dryer')
electric_oven = redd_data.buildings[1].elec.select_using_appliances(type='electric oven')



'''
Display total energy per submeter group (so we can rank by total meter energy)

'''

'''
print("fridge: "+str(fridge.total_energy()))
print("dishwasher: "+str(dish_washer.total_energy()))
print("sockets: "+str(sockets.total_energy()))
print("lights: "+str(lights.total_energy()))
print("unknowns:"+str(unknowns.total_energy()))
print("microwave: "+str(microwave.total_energy()))
print("electric space heater: "+str(electric_space_heater.total_energy()))
print("electric stove: "+str(electric_stove.total_energy()))
print("washer dryer: "+str(washer_dryer.total_energy()))
print("electric oven: "+str(electric_oven.total_energy()))
'''

'''
From inspection total energy usage rankings:
1) lights
2) sockets
3) fridge
4) washer dryer
5) dish washer
6) microwave
7) electric oven
8) unknown
9) electric stove
10) electric space heater

'''

print("Creating Training Sets...")
'''
Training set 1: Top 2 Energy Appliances
'''
training_set1 = lights.union(sockets)

'''
Training Set 2: Top 3 Energy Appliances
'''
training_set2 = training_set1.union(fridge)

'''
Training Set 3: Top 4 Energy Appliances
'''
training_set3 = training_set2.union(washer_dryer)

'''
Training Set 4: Top 5 Energy Appliances
'''
training_set4 = training_set3.union(dish_washer)

'''
Training Set 5: Top 6 Energy Appliances
'''
training_set5 = training_set4.union(microwave)

'''
Training Set 6: Top 7 Energy Appliances
'''
training_set6 = training_set5.union(electric_oven)

'''
Training Set 7: Top 8 Energy Appliances
'''
training_set7 = training_set6.union(unknowns)

'''
Training Set 8: Top 9 Energy Appliances
'''
training_set8 = training_set7.union(electric_stove)

'''
Training Set 9: Top 10 Energy Appliances
'''
training_set9 = training_set8.union(electric_space_heater)

print("Training sets created!")

'''
Create 9 instances of disaggregation model and train each with
unique training set.

'''
print("Training disaggregation algorithms...")

co1 = CombinatorialOptimisation()
co2 = CombinatorialOptimisation()
co3 = CombinatorialOptimisation()
co4 = CombinatorialOptimisation()
co5 = CombinatorialOptimisation()
co6 = CombinatorialOptimisation()
co7 = CombinatorialOptimisation()
co8 = CombinatorialOptimisation()
co9 = CombinatorialOptimisation()

co1.train(training_set1)
print("set 1 trained")
co2.train(training_set2)
print("set 2 trained")
co3.train(training_set3)
print("set 3 trained")
co4.train(training_set4)
print("set 4 trained")
co5.train(training_set5)
print("set 5 trained")
co6.train(training_set6)
print("set 6 trained")
co7.train(training_set7)
print("set 7 trained")
co8.train(training_set8)
print("set 8 trained")
co9.train(training_set9)
print("set 9 trained")

print("Algorithms trained!")

'''
Create 9 output files to hold disaggregated data.

'''

print("Creating output files...")

outData1 = HDFDataStore("C:/NILM/Data/Model_Train/output1.h5",'w')
outData2 = HDFDataStore("C:/NILM/Data/Model_Train/output2.h5",'w')
outData3 = HDFDataStore("C:/NILM/Data/Model_Train/output3.h5",'w')
outData4 = HDFDataStore("C:/NILM/Data/Model_Train/output4.h5",'w')
outData5 = HDFDataStore("C:/NILM/Data/Model_Train/output5.h5",'w')
outData6 = HDFDataStore("C:/NILM/Data/Model_Train/output6.h5",'w')
outData7 = HDFDataStore("C:/NILM/Data/Model_Train/output7.h5",'w')
outData8 = HDFDataStore("C:/NILM/Data/Model_Train/output8.h5",'w')
outData9 = HDFDataStore("C:/NILM/Data/Model_Train/output9.h5",'w')

print("output files created!")

'''
Disaggregate building 1 data using each training set
'''

print("Disaggregating building 1 mains using each trained model...")

#set building 1 mains
b1_mains = redd_data.buildings[1].elec.mains()


co1.disaggregate(b1_mains, outData1)
print("mains disaggregated with set 1 model")
co2.disaggregate(b1_mains, outData2)
print("mains disaggregated with set 2 model")
co3.disaggregate(b1_mains, outData3)
print("mains disaggregated with set 3 model")
co4.disaggregate(b1_mains, outData4)
print("mains disaggregated with set 4 model")
co5.disaggregate(b1_mains, outData5)
print("mains disaggregated with set 5 model")
co6.disaggregate(b1_mains, outData6)
print("mains disaggregated with set 6 model")
co7.disaggregate(b1_mains, outData7)
print("mains disaggregated with set 7 model")
co8.disaggregate(b1_mains, outData8)
print("mains disaggregated with set 8 model")
co9.disaggregate(b1_mains, outData9)
print("mains disaggregated with set 9 model")

print("Building 1 mains sucessfully disaggregated!")


#Close dataset dataStore
redd_data.store.close()
outData1.close()
outData2.close()
outData3.close()
outData4.close()
outData5.close()
outData6.close()
outData7.close()
outData8.close()
outData9.close()