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
from nilmtk.metrics import *
import warnings

#supress warnings to users console
warnings.filterwarnings("ignore")

#create numbered dictonary of meter# to appliance name


t1 = "2011-05-1 1:00"
t2 = "2011-05-01 12:00"	

#declate key_map object
km = Key_Map(1)

#load original redd data
print("Loading REDD and Disaggregated Datasets...")
redd_data = DataSet("C:/NILM/Data/redd_data.h5")
out1 = DataSet("C:/NILM/Data/Model_Train/output1.h5")
out2 = DataSet("C:/NILM/Data/Model_Train/output2.h5")
out3 = DataSet("C:/NILM/Data/Model_Train/output3.h5")
out4 = DataSet("C:/NILM/Data/Model_Train/output4.h5")
out5 = DataSet("C:/NILM/Data/Model_Train/output5.h5")
out6 = DataSet("C:/NILM/Data/Model_Train/output6.h5")
out7 = DataSet("C:/NILM/Data/Model_Train/output7.h5")
out8 = DataSet("C:/NILM/Data/Model_Train/output8.h5")
out9 = DataSet("C:/NILM/Data/Model_Train/output9.h5")
print("DataSets loaded!")


#print out meters found in each set
'''
print(redd_data.buildings[1].elec.map_meter_instances_to_appliance_ids())
print(out1.buildings[1].elec.map_meter_instances_to_appliance_ids())
print(out2.buildings[1].elec.map_meter_instances_to_appliance_ids())
print(out3.buildings[1].elec.map_meter_instances_to_appliance_ids())
print(out4.buildings[1].elec.map_meter_instances_to_appliance_ids())
print(out5.buildings[1].elec.map_meter_instances_to_appliance_ids())
print(out6.buildings[1].elec.map_meter_instances_to_appliance_ids())
print(out7.buildings[1].elec.map_meter_instances_to_appliance_ids())
print(out8.buildings[1].elec.map_meter_instances_to_appliance_ids())
print(out9.buildings[1].elec.map_meter_instances_to_appliance_ids())
'''

#plot appliance's actual fingerprint vs predicted fingerprint
'''
plot_series(out9.store.store.get(km.get_key('fridge'))[t1:t2])
plot_series(redd_data.store.store.get(km.get_key('fridge'))[t1:t2])
plt.show()
'''

'''
Metrics analysis: check each training model's performance using built in Metrics

'''

f1 = open('C:/NILM/Data/Model_Train/Error_Output/AS_Error/as_er1.txt','w')
f2 = open('C:/NILM/Data/Model_Train/Error_Output/AS_Error/as_er2.txt','w')
f3 = open('C:/NILM/Data/Model_Train/Error_Output/AS_Error/as_er3.txt','w')
f4 = open('C:/NILM/Data/Model_Train/Error_Output/AS_Error/as_er4.txt','w')
f5 = open('C:/NILM/Data/Model_Train/Error_Output/AS_Error/as_er5.txt','w')
f6 = open('C:/NILM/Data/Model_Train/Error_Output/AS_Error/as_er6.txt','w')
f7 = open('C:/NILM/Data/Model_Train/Error_Output/AS_Error/as_er7.txt','w')
f8 = open('C:/NILM/Data/Model_Train/Error_Output/AS_Error/as_er8.txt','w')
f9 = open('C:/NILM/Data/Model_Train/Error_Output/AS_Error/as_er9.txt','w')


print("\nComputing error in assigned energy for each disaggregation model...")
as_error1 = error_in_assigned_energy(out1.buildings[1].elec, redd_data.buildings[1].elec)
f1.write(repr(as_error1))
f1.close()
print("Set 1 metric result written to file.")
as_error2 = error_in_assigned_energy(out2.buildings[1].elec, redd_data.buildings[1].elec)
f2.write(repr(as_error2))
f2.close()
print("Set 2 metric result written to file.")
as_error3 = error_in_assigned_energy(out3.buildings[1].elec, redd_data.buildings[1].elec)
f3.write(repr(as_error3))
f3.close()
print("Set 3 metric result written to file.")
as_error4 = error_in_assigned_energy(out4.buildings[1].elec, redd_data.buildings[1].elec)
f4.write(repr(as_error4))
f4.close()
print("Set 4 metric result written to file.")
as_error5 = error_in_assigned_energy(out5.buildings[1].elec, redd_data.buildings[1].elec)
f5.write(repr(as_error5))
f5.close()
print("Set 5 metric result written to file.")
as_error6 = error_in_assigned_energy(out6.buildings[1].elec, redd_data.buildings[1].elec)
f6.write(repr(as_error6))
f6.close()
print("Set 6 metric result written to file.")
as_error7 = error_in_assigned_energy(out7.buildings[1].elec, redd_data.buildings[1].elec)
f7.write(repr(as_error7))
f7.close()
print("Set 7 metric result written to file.")
as_error8 = error_in_assigned_energy(out8.buildings[1].elec, redd_data.buildings[1].elec)
f8.write(repr(as_error8))
f8.close()
print("Set 8 metric result written to file.")
as_error9 = error_in_assigned_energy(out9.buildings[1].elec, redd_data.buildings[1].elec)
f9.write(repr(as_error9))
f9.close()
print("Set 9 metric result written to file.")
print ("Error Computed!")


f1 = open('C:/NILM/Data/Model_Train/Error_Output/RMS_Error/rms_er1.txt','w')
f2 = open('C:/NILM/Data/Model_Train/Error_Output/RMS_Error/rms_er2.txt','w')
f3 = open('C:/NILM/Data/Model_Train/Error_Output/RMS_Error/rms_er3.txt','w')
f4 = open('C:/NILM/Data/Model_Train/Error_Output/RMS_Error/rms_er4.txt','w')
f5 = open('C:/NILM/Data/Model_Train/Error_Output/RMS_Error/rms_er5.txt','w')
f6 = open('C:/NILM/Data/Model_Train/Error_Output/RMS_Error/rms_er6.txt','w')
f7 = open('C:/NILM/Data/Model_Train/Error_Output/RMS_Error/rms_er7.txt','w')
f8 = open('C:/NILM/Data/Model_Train/Error_Output/RMS_Error/rms_er8.txt','w')
f9 = open('C:/NILM/Data/Model_Train/Error_Output/RMS_Error/rms_er9.txt','w')


print("\nComputing RMS error in power for each disaggregation model...")
rms_error1 = rms_error_power(out1.buildings[1].elec, redd_data.buildings[1].elec)
f1.write(repr(rms_error1))
f1.close()
print("Set 1 metric result written to file.")
rms_error2 = rms_error_power(out2.buildings[1].elec, redd_data.buildings[1].elec)
f2.write(repr(rms_error2))
f2.close()
print("Set 2 metric result written to file.")
rms_error3 = rms_error_power(out3.buildings[1].elec, redd_data.buildings[1].elec)
f3.write(repr(rms_error3))
f3.close()
print("Set 3 metric result written to file.")
rms_error4 = rms_error_power(out4.buildings[1].elec, redd_data.buildings[1].elec)
f4.write(repr(rms_error4))
f4.close()
print("Set 4 metric result written to file.")
rms_error5 = rms_error_power(out5.buildings[1].elec, redd_data.buildings[1].elec)
f5.write(repr(rms_error5))
f5.close()
print("Set 5 metric result written to file.")
rms_error6 = rms_error_power(out6.buildings[1].elec, redd_data.buildings[1].elec)
f6.write(repr(rms_error6))
f6.close()
print("Set 6 metric result written to file.")
rms_error7 = rms_error_power(out7.buildings[1].elec, redd_data.buildings[1].elec)
f7.write(repr(rms_error7))
f7.close()
print("Set 7 metric result written to file.")
rms_error8 = rms_error_power(out8.buildings[1].elec, redd_data.buildings[1].elec)
f8.write(repr(rms_error8))
f8.close()
print("Set 8 metric result written to file.")
rms_error9 = rms_error_power(out9.buildings[1].elec, redd_data.buildings[1].elec)
f9.write(repr(rms_error9))
f9.close()
print("Set 9 metric result written to file.")
print("Error Computed!")


f1 = open('C:/NILM/Data/Model_Train/Error_Output/F1_Score/f1score1.txt','w')
f2 = open('C:/NILM/Data/Model_Train/Error_Output/F1_Score/f1score2.txt','w')
f3 = open('C:/NILM/Data/Model_Train/Error_Output/F1_Score/f1score3.txt','w')
f4 = open('C:/NILM/Data/Model_Train/Error_Output/F1_Score/f1score4.txt','w')
f5 = open('C:/NILM/Data/Model_Train/Error_Output/F1_Score/f1score5.txt','w')
f6 = open('C:/NILM/Data/Model_Train/Error_Output/F1_Score/f1score6.txt','w')
f7 = open('C:/NILM/Data/Model_Train/Error_Output/F1_Score/f1score7.txt','w')
f8 = open('C:/NILM/Data/Model_Train/Error_Output/F1_Score/f1score8.txt','w')
f9 = open('C:/NILM/Data/Model_Train/Error_Output/F1_Score/f1score9.txt','w')


print("\nComputing f1_score in power for each disaggregation model...")
f1_score1 = f1_score(out1.buildings[1].elec, redd_data.buildings[1].elec)
f1.write(repr(f1_score1))
f1.close()
print("Set 1 metric result written to file.")
f1_score2 = f1_score(out2.buildings[1].elec, redd_data.buildings[1].elec)
f2.write(repr(f1_score2))
f2.close()
print("Set 2 metric result written to file.")
f1_score3 = f1_score(out3.buildings[1].elec, redd_data.buildings[1].elec)
f3.write(repr(f1_score3))
f3.close()
print("Set 3 metric result written to file.")
f1_score4 = f1_score(out4.buildings[1].elec, redd_data.buildings[1].elec)
f4.write(repr(f1_score4))
f4.close()
print("Set 4 metric result written to file.")
f1_score5 = f1_score(out5.buildings[1].elec, redd_data.buildings[1].elec)
f5.write(repr(f1_score5))
f5.close()
print("Set 5 metric result written to file.")
f1_score6 = f1_score(out6.buildings[1].elec, redd_data.buildings[1].elec)
f6.write(repr(f1_score6))
f6.close()
print("Set 6 metric result written to file.")
f1_score7 = f1_score(out7.buildings[1].elec, redd_data.buildings[1].elec)
f7.write(repr(f1_score7))
f7.close()
print("Set 7 metric result written to file.")
f1_score8 = f1_score(out8.buildings[1].elec, redd_data.buildings[1].elec)
f8.write(repr(f1_score8))
f8.close()
print("Set 8 metric result written to file.")
f1_score9 = f1_score(out9.buildings[1].elec, redd_data.buildings[1].elec)
f9.write(repr(f1_score9))
f9.close()
print("Set 9 metric result written to file.")
print("Error Computed!")

print("Calculating fraction of correct assignment per training group...")

f1 = open('C:/NILM/Data/Model_Train/Error_Output/Correct_Fraction/cor_frac1.txt','w')
f2 = open('C:/NILM/Data/Model_Train/Error_Output/Correct_Fraction/cor_frac2.txt','w')
f3 = open('C:/NILM/Data/Model_Train/Error_Output/Correct_Fraction/cor_frac3.txt','w')
f4 = open('C:/NILM/Data/Model_Train/Error_Output/Correct_Fraction/cor_frac4.txt','w')
f5 = open('C:/NILM/Data/Model_Train/Error_Output/Correct_Fraction/cor_frac5.txt','w')
f6 = open('C:/NILM/Data/Model_Train/Error_Output/Correct_Fraction/cor_frac6.txt','w')
f7 = open('C:/NILM/Data/Model_Train/Error_Output/Correct_Fraction/cor_frac7.txt','w')
f8 = open('C:/NILM/Data/Model_Train/Error_Output/Correct_Fraction/cor_frac8.txt','w')
f9 = open('C:/NILM/Data/Model_Train/Error_Output/Correct_Fraction/cor_frac9.txt','w')

print("Calucating for set 1")
cor1 = fraction_energy_assigned_correctly(out1.buildings[1].elec, redd_data.buildings[1].elec)
f1.write(repr(cor1))
f1.close
print("Calucating for set 2")
cor2 = fraction_energy_assigned_correctly(out2.buildings[1].elec, redd_data.buildings[1].elec)
f2.write(repr(cor2))
f2.close
print("Calucating for set 3")
cor3 = fraction_energy_assigned_correctly(out3.buildings[1].elec, redd_data.buildings[1].elec)
f3.write(repr(cor3))
f3.close
print("Calucating for set 4")
cor4 = fraction_energy_assigned_correctly(out4.buildings[1].elec, redd_data.buildings[1].elec)
f4.write(repr(cor4))
f4.close
print("Calucating for set 5")
cor5 = fraction_energy_assigned_correctly(out5.buildings[1].elec, redd_data.buildings[1].elec)
f5.write(repr(cor5))
f5.close
print("Calucating for set 6")
cor6 = fraction_energy_assigned_correctly(out6.buildings[1].elec, redd_data.buildings[1].elec)
f6.write(repr(cor6))
f6.close
print("Calucating for set 7")
cor7 = fraction_energy_assigned_correctly(out7.buildings[1].elec, redd_data.buildings[1].elec)
f7.write(repr(cor7))
f7.close
print("Calucating for set 8")
cor8 = fraction_energy_assigned_correctly(out8.buildings[1].elec, redd_data.buildings[1].elec)
f8.write(repr(cor8))
f8.close
print("Calucating for set 9")
cor9 = fraction_energy_assigned_correctly(out9.buildings[1].elec, redd_data.buildings[1].elec)
f9.write(repr(cor9))
f9.close

print("Calculations finished.")



#close files
print("Closing DataStore files.")
redd_data.store.close()
out1.store.close()
out2.store.close()
out3.store.close()
out4.store.close()
out5.store.close()
out6.store.close()
out7.store.close()
out8.store.close()
out9.store.close()