#Author: Michael Milicevich
# Script defines and trains the machine learning module
# Used for finding optimal training set (for REDD)

from __future__ import print_function, division
from nilmtk import HDFDataStore, DataSet, TimeFrame, MeterGroup
from nilmtk.disaggregate import CombinatorialOptimisation
import pandas as pd
import matplotlib.pyplot as plt
from nilmtk.plots import plot_series
from key_map import *
import warnings
import time

#supress warnings to users console
warnings.filterwarnings("ignore")

#declare key map object set to building 1
km = Key_Map(1)

print("Loading Dataset...")
#declare REDD datastore
redd_data = DataSet("C:/NILM/Data/REDD/redd.h5")
#redd_data.store.window = TimeFrame(start = "2011-05-01 00:00:00+00:00", end = "2011-05-01 12:00:00+00:00")
print("Dataset Loaded!")


'''
Preprocess the data so that 

'''




'''
Unable to get select_top_k() function working so must manually declare each appliance in its own metergroup in
order to get combinations for testing different training groups.

'''

print ("Creating training groups...")

fridge = redd_data.buildings[1].elec.select_using_appliances(type='fridge')
dish_washer = redd_data.buildings[1].elec.select_using_appliances(type='dish washer')

sockets1 = MeterGroup([redd_data.buildings[1].elec.__getitem__(7)])
sockets2 = MeterGroup([redd_data.buildings[1].elec.__getitem__(8)])
sockets3 = MeterGroup([redd_data.buildings[1].elec.__getitem__(15)])
sockets4 = MeterGroup([redd_data.buildings[1].elec.__getitem__(16)])
lights1 = MeterGroup([redd_data.buildings[1].elec.__getitem__(9)])
lights2 = MeterGroup([redd_data.buildings[1].elec.__getitem__(17)])
lights3 = MeterGroup([redd_data.buildings[1].elec.__getitem__(18)])
unknown1 = MeterGroup([redd_data.buildings[1].elec.__getitem__(12) ])
unknown2 = MeterGroup([redd_data.buildings[1].elec.__getitem__(19)])
microwave = redd_data.buildings[1].elec.select_using_appliances(type='microwave')
electric_space_heater = redd_data.buildings[1].elec.select_using_appliances(type='electric space heater')
electric_stove = redd_data.buildings[1].elec.select_using_appliances(type='electric stove')
washer_dryer = redd_data.buildings[1].elec.select_using_appliances(type='washer dryer')
electric_oven = redd_data.buildings[1].elec.select_using_appliances(type='electric oven')


'''
Print out total energy for each sub-meter, will use this to manually determine top-k appliances

'''
'''
print("fridge: "+str(fridge.total_energy()))
print("dishwasher: "+str(dish_washer.total_energy()))
print("socket1: "+str(sockets1.total_energy()))
print("socket2: "+str(sockets2.total_energy()))
print("socket3: "+str(sockets3.total_energy()))
print("socket4: "+str(sockets4.total_energy()))
print("lights1: "+str(lights1.total_energy()))
print("lights2: "+str(lights2.total_energy()))
print("lights3: "+str(lights3.total_energy()))
print("unknown1: "+str(unknown1.total_energy()))
print("unknown2: "+str(unknown2.total_energy()))
print("microwave: "+str(microwave.total_energy()))
print("electric_space_heater: "+str(electric_space_heater.total_energy()))
print("electric_stove: "+str(electric_stove.total_energy()))
print("washer_dryer: "+str(washer_dryer.total_energy()))
print("electric_oven: "+str(electric_oven.total_energy()))

'''
'''
From Inspection top_k appliances are:

1) Fridge
2) Washer Dryer
3) Lights 1
4) Sockets 2
5) Dish Washer
6) Lights 2
7) Microwave
8) Sockets 1
9) Lights 3
10) Electric oven
11) Unknown 1
12) Sockets 3
13) Sockets 4
14) Electric stove
15) Electric Space heater
16) Unknown 2

'''

'''
Manually create 15 sets, each one adding one more appliance, starting with the largest
two energy consuming appliances.

'''


training_set1 = fridge.union(washer_dryer)
training_set2 = training_set1.union(lights1)
training_set3 = training_set2.union(sockets2)
training_set4 = training_set3.union(dish_washer)
training_set5 = training_set4.union(lights2)
training_set6 = training_set5.union(microwave)
training_set7 = training_set6.union(sockets1)
training_set8 = training_set7.union(lights3)
training_set9 = training_set8.union(electric_oven)
training_set10 = training_set9.union(unknown1)
training_set11 = training_set10.union(sockets3)
training_set12 = training_set11.union(sockets4)
training_set13 = training_set12.union(electric_stove)
training_set14 = training_set13.union(electric_space_heater)
training_set15 = training_set14.union(unknown2)


print("Training groups sucessfully created!")


co1 = CombinatorialOptimisation()
co2 = CombinatorialOptimisation()
co3 = CombinatorialOptimisation()
co4 = CombinatorialOptimisation()
co5 = CombinatorialOptimisation()
co6 = CombinatorialOptimisation()
co7 = CombinatorialOptimisation()
co8 = CombinatorialOptimisation()
co9 = CombinatorialOptimisation()
co10 = CombinatorialOptimisation()
co11 = CombinatorialOptimisation()
co12 = CombinatorialOptimisation()
co13 = CombinatorialOptimisation()
co14 = CombinatorialOptimisation()
co15 = CombinatorialOptimisation()


print("Training algorithms....")

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
co10.train(training_set10)
print("set 10 trained")
co11.train(training_set11)
print("set 11 trained")
co12.train(training_set12)
print("set 12 trained")
co13.train(training_set13)
print("set 13 trained")
co14.train(training_set14)
print("set 14 trained")
co15.train(training_set15)
print("set 15 trained")


print("Algorithms sucessfully trained!")


'''
Create 15 output files to hold disaggregated data.

'''

print("Creating output files...")

# outData1 = HDFDataStore("C:/NILM/Data/Model_Train/output1.h5",'w')
# outData2 = HDFDataStore("C:/NILM/Data/Model_Train/output2.h5",'w')
# outData3 = HDFDataStore("C:/NILM/Data/Model_Train/output3.h5",'w')
# outData4 = HDFDataStore("C:/NILM/Data/Model_Train/output4.h5",'w')
# outData5 = HDFDataStore("C:/NILM/Data/Model_Train/output5.h5",'w')
# outData6 = HDFDataStore("C:/NILM/Data/Model_Train/output6.h5",'w')
# outData7 = HDFDataStore("C:/NILM/Data/Model_Train/output7.h5",'w')
# outData8 = HDFDataStore("C:/NILM/Data/Model_Train/output8.h5",'w')
# outData9 = HDFDataStore("C:/NILM/Data/Model_Train/output9.h5",'w')
# outData10 = HDFDataStore("C:/NILM/Data/Model_Train/output10.h5",'w')
# outData11 = HDFDataStore("C:/NILM/Data/Model_Train/output11.h5",'w')
# outData12 = HDFDataStore("C:/NILM/Data/Model_Train/output12.h5",'w')
# outData13 = HDFDataStore("C:/NILM/Data/Model_Train/output13.h5",'w')
# outData14 = HDFDataStore("C:/NILM/Data/Model_Train/output14.h5",'w')
# outData15 = HDFDataStore("C:/NILM/Data/Model_Train/output15.h5",'w')

# print("output files created!")


# '''
# Disaggregate building 1 data using each training set
# '''

# print("Disaggregating building 1 mains using each trained model...")

# #set building 1 mains
# b1_mains = redd_data.buildings[1].elec.mains()


# current_time = time.time()
# co1.disaggregate(b1_mains, outData1)
# time1 = time.time() - current_time
# print("mains disaggregated with set 1 model")

# current_time = time.time()
# co2.disaggregate(b1_mains, outData2)
# time2 =time.time() - current_time
# print("mains disaggregated with set 2 model")

# current_time = time.time()
# co3.disaggregate(b1_mains, outData3)
# time3 =time.time() - current_time
# print("mains disaggregated with set 3 model")

# current_time = time.time()
# co4.disaggregate(b1_mains, outData4)
# time4 =time.time() - current_time
# print("mains disaggregated with set 4 model")

# current_time = time.time()
# co5.disaggregate(b1_mains, outData5)
# time5 =time.time() - current_time
# print("mains disaggregated with set 5 model")

# current_time = time.time()
# co6.disaggregate(b1_mains, outData6)
# time6 =time.time() - current_time
# print("mains disaggregated with set 6 model")

# current_time = time.time()
# co7.disaggregate(b1_mains, outData7)
# time7 =time.time() - current_time
# print("mains disaggregated with set 7 model")

# current_time = time.time()
# co8.disaggregate(b1_mains, outData8)
# time8 =time.time() - current_time
# print("mains disaggregated with set 8 model")

# current_time = time.time()
# co9.disaggregate(b1_mains, outData9)
# time9 =time.time() - current_time
# print("mains disaggregated with set 9 model")

# current_time = time.time()
# co10.disaggregate(b1_mains, outData10)
# time10 =time.time() - current_time
# print("mains disaggregated with set 10 model")

# current_time = time.time()
# co11.disaggregate(b1_mains, outData11)
# time11 =time.time() - current_time
# print("mains disaggregated with set 11 model")

# current_time = time.time()
# co12.disaggregate(b1_mains, outData12)
# time12 =time.time() - current_time
# print("mains disaggregated with set 12 model")

# current_time = time.time()
# co13.disaggregate(b1_mains, outData13)
# time13 =time.time() - current_time
# print("mains disaggregated with set 13 model")

# current_time = time.time()
# co14.disaggregate(b1_mains, outData14)
# time14 =time.time() - current_time
# print("mains disaggregated with set 13 model")

# current_time = time.time()
# co15.disaggregate(b1_mains, outData15)
# time15 =time.time() - current_time
# print("mains disaggregated with set 15 model")

# print("Building 1 mains sucessfully disaggregated!")

# print("Writing disaggregation timing to file..")

# f = open("C:/NILM/Data/Model_Train/Error_Output/Timing/disag_time.txt",'w')
# f.write(str(time1))
# f.write("	")

# f.write(str(time2))
# f.write("	")

# f.write(str(time3))
# f.write("	")

# f.write(str(time4))
# f.write("	")

# f.write(str(time5))
# f.write("	")

# f.write(str(time6))
# f.write("	")

# f.write(str(time7))
# f.write("	")

# f.write(str(time8))
# f.write("	")

# f.write(str(time9))
# f.write("	")

# f.write(str(time10))
# f.write("	")

# f.write(str(time11))
# f.write("	")

# f.write(str(time12))
# f.write("	")

# f.write(str(time13))
# f.write("	")

# f.write(str(time14))
# f.write("	")

# f.write(str(time15))
# f.close()

# print("Timing sucessfully written to file.")

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
outData10.close()
outData11.close()
outData12.close()
outData13.close()
outData14.close()
outData15.close()