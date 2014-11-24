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
out10 = DataSet("C:/NILM/Data/Model_Train/output10.h5")
out11 = DataSet("C:/NILM/Data/Model_Train/output11.h5")
out12 = DataSet("C:/NILM/Data/Model_Train/output12.h5")
out13 = DataSet("C:/NILM/Data/Model_Train/output13.h5")
out14 = DataSet("C:/NILM/Data/Model_Train/output14.h5")
out15 = DataSet("C:/NILM/Data/Model_Train/output15.h5")
print("DataSets loaded!")

'''
Metrics Analysis
'''

#open files to output results
f1 = open('C:/NILM/Data/Model_Train/Error_Output/NAS_Error/nas_er1.txt','w')
f2 = open('C:/NILM/Data/Model_Train/Error_Output/NAS_Error/nas_er2.txt','w')
f3 = open('C:/NILM/Data/Model_Train/Error_Output/NAS_Error/nas_er3.txt','w')
f4 = open('C:/NILM/Data/Model_Train/Error_Output/NAS_Error/nas_er4.txt','w')
f5 = open('C:/NILM/Data/Model_Train/Error_Output/NAS_Error/nas_er5.txt','w')
f6 = open('C:/NILM/Data/Model_Train/Error_Output/NAS_Error/nas_er6.txt','w')
f7 = open('C:/NILM/Data/Model_Train/Error_Output/NAS_Error/nas_er7.txt','w')
f8 = open('C:/NILM/Data/Model_Train/Error_Output/NAS_Error/nas_er8.txt','w')
f9 = open('C:/NILM/Data/Model_Train/Error_Output/NAS_Error/nas_er9.txt','w')
f10 = open('C:/NILM/Data/Model_Train/Error_Output/NAS_Error/nas_er10.txt','w')
f11 = open('C:/NILM/Data/Model_Train/Error_Output/NAS_Error/nas_er11.txt','w')
f12 = open('C:/NILM/Data/Model_Train/Error_Output/NAS_Error/nas_er12.txt','w')
f13 = open('C:/NILM/Data/Model_Train/Error_Output/NAS_Error/nas_er13.txt','w')
f14 = open('C:/NILM/Data/Model_Train/Error_Output/NAS_Error/nas_er14.txt','w')
f15 = open('C:/NILM/Data/Model_Train/Error_Output/NAS_Error/nas_er15.txt','w')


print("\nComputing normalized error in assigned energy for each disaggregation model...")

f1.write(repr(mean_normalized_error_power(out1.buildings[1].elec, redd_data.buildings[1].elec)))
f1.close()
print("Set 1 metric result written to file.")

f2.write(repr(mean_normalized_error_power(out2.buildings[1].elec, redd_data.buildings[1].elec)))
f2.close()
print("Set 2 metric result written to file.")

f3.write(repr(mean_normalized_error_power(out3.buildings[1].elec, redd_data.buildings[1].elec)))
f3.close()
print("Set 3 metric result written to file.")

f4.write(repr(mean_normalized_error_power(out4.buildings[1].elec, redd_data.buildings[1].elec)))
f4.close()
print("Set 4 metric result written to file.")

f5.write(repr(mean_normalized_error_power(out5.buildings[1].elec, redd_data.buildings[1].elec)))
f5.close()
print("Set 5 metric result written to file.")

f6.write(repr(mean_normalized_error_power(out6.buildings[1].elec, redd_data.buildings[1].elec)))
f6.close()
print("Set 6 metric result written to file.")

f7.write(repr(mean_normalized_error_power(out7.buildings[1].elec, redd_data.buildings[1].elec)))
f7.close()
print("Set 7 metric result written to file.")

f8.write(repr(mean_normalized_error_power(out8.buildings[1].elec, redd_data.buildings[1].elec)))
f8.close()
print("Set 8 metric result written to file.")

f9.write(repr(mean_normalized_error_power(out9.buildings[1].elec, redd_data.buildings[1].elec)))
f9.close()
print("Set 9 metric result written to file.")

f10.write(repr(mean_normalized_error_power(out10.buildings[1].elec, redd_data.buildings[1].elec)))
f10.close()
print("Set 10 metric result written to file.")

f11.write(repr(mean_normalized_error_power(out11.buildings[1].elec, redd_data.buildings[1].elec)))
f11.close()
print("Set 11 metric result written to file.")

f12.write(repr(mean_normalized_error_power(out12.buildings[1].elec, redd_data.buildings[1].elec)))
f12.close()
print("Set 12 metric result written to file.")

f13.write(repr(mean_normalized_error_power(out13.buildings[1].elec, redd_data.buildings[1].elec)))
f13.close()
print("Set 13 metric result written to file.")

f14.write(repr(mean_normalized_error_power(out14.buildings[1].elec, redd_data.buildings[1].elec)))
f14.close()
print("Set 14 metric result written to file.")

f15.write(repr(mean_normalized_error_power(out15.buildings[1].elec, redd_data.buildings[1].elec)))
f15.close()
print("Set 15 metric result written to file.")
print("All Mean Normalized Error results written to file.")


f1 = open('C:/NILM/Data/Model_Train/Error_Output/RMS_Error/rms_er1.txt','w')
f2 = open('C:/NILM/Data/Model_Train/Error_Output/RMS_Error/rms_er2.txt','w')
f3 = open('C:/NILM/Data/Model_Train/Error_Output/RMS_Error/rms_er3.txt','w')
f4 = open('C:/NILM/Data/Model_Train/Error_Output/RMS_Error/rms_er4.txt','w')
f5 = open('C:/NILM/Data/Model_Train/Error_Output/RMS_Error/rms_er5.txt','w')
f6 = open('C:/NILM/Data/Model_Train/Error_Output/RMS_Error/rms_er6.txt','w')
f7 = open('C:/NILM/Data/Model_Train/Error_Output/RMS_Error/rms_er7.txt','w')
f8 = open('C:/NILM/Data/Model_Train/Error_Output/RMS_Error/rms_er8.txt','w')
f9 = open('C:/NILM/Data/Model_Train/Error_Output/RMS_Error/rms_er9.txt','w')
f10 = open('C:/NILM/Data/Model_Train/Error_Output/RMS_Error/rms_er10.txt','w')
f11 = open('C:/NILM/Data/Model_Train/Error_Output/RMS_Error/rms_er11.txt','w')
f12 = open('C:/NILM/Data/Model_Train/Error_Output/RMS_Error/rms_er12.txt','w')
f13 = open('C:/NILM/Data/Model_Train/Error_Output/RMS_Error/rms_er13.txt','w')
f14 = open('C:/NILM/Data/Model_Train/Error_Output/RMS_Error/rms_er14.txt','w')
f15 = open('C:/NILM/Data/Model_Train/Error_Output/RMS_Error/rms_er15.txt','w')

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

f10.write(repr(rms_error_power(out10.buildings[1].elec, redd_data.buildings[1].elec)))
f10.close()
print("Set 10 metric result written to file.")

f11.write(repr(rms_error_power(out11.buildings[1].elec, redd_data.buildings[1].elec)))
f11.close()
print("Set 11 metric result written to file.")

f12.write(repr(rms_error_power(out12.buildings[1].elec, redd_data.buildings[1].elec)))
f12.close()
print("Set 12 metric result written to file.")

f13.write(repr(rms_error_power(out13.buildings[1].elec, redd_data.buildings[1].elec)))
f13.close()
print("Set 13 metric result written to file.")

f14.write(repr(rms_error_power(out14.buildings[1].elec, redd_data.buildings[1].elec)))
f14.close()
print("Set 14 metric result written to file.")

f15.write(repr(rms_error_power(out15.buildings[1].elec, redd_data.buildings[1].elec)))
f15.close()
print("Set 15 metric result written to file.")

print("All RMS Error results written to file.")


f1 = open('C:/NILM/Data/Model_Train/Error_Output/F1_Score/f1score1.txt','w')
f2 = open('C:/NILM/Data/Model_Train/Error_Output/F1_Score/f1score2.txt','w')
f3 = open('C:/NILM/Data/Model_Train/Error_Output/F1_Score/f1score3.txt','w')
f4 = open('C:/NILM/Data/Model_Train/Error_Output/F1_Score/f1score4.txt','w')
f5 = open('C:/NILM/Data/Model_Train/Error_Output/F1_Score/f1score5.txt','w')
f6 = open('C:/NILM/Data/Model_Train/Error_Output/F1_Score/f1score6.txt','w')
f7 = open('C:/NILM/Data/Model_Train/Error_Output/F1_Score/f1score7.txt','w')
f8 = open('C:/NILM/Data/Model_Train/Error_Output/F1_Score/f1score8.txt','w')
f9 = open('C:/NILM/Data/Model_Train/Error_Output/F1_Score/f1score9.txt','w')
f10 = open('C:/NILM/Data/Model_Train/Error_Output/F1_Score/f1score10.txt','w')
f11 = open('C:/NILM/Data/Model_Train/Error_Output/F1_Score/f1score11.txt','w')
f12 = open('C:/NILM/Data/Model_Train/Error_Output/F1_Score/f1score12.txt','w')
f13 = open('C:/NILM/Data/Model_Train/Error_Output/F1_Score/f1score13.txt','w')
f14 = open('C:/NILM/Data/Model_Train/Error_Output/F1_Score/f1score14.txt','w')
f15 = open('C:/NILM/Data/Model_Train/Error_Output/F1_Score/f1score15.txt','w')


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

f10.write(repr(f1_score(out10.buildings[1].elec, redd_data.buildings[1].elec)))
f10.close()
print("Set 10 metric result written to file.")

f11.write(repr(f1_score(out11.buildings[1].elec, redd_data.buildings[1].elec)))
f11.close()
print("Set 11 metric result written to file.")

f12.write(repr(f1_score(out12.buildings[1].elec, redd_data.buildings[1].elec)))
f12.close()
print("Set 12 metric result written to file.")

f13.write(repr(f1_score(out13.buildings[1].elec, redd_data.buildings[1].elec)))
f13.close()
print("Set 13 metric result written to file.")

f14.write(repr(f1_score(out14.buildings[1].elec, redd_data.buildings[1].elec)))
f14.close()
print("Set 14 metric result written to file.")

f15.write(repr(f1_score(out15.buildings[1].elec, redd_data.buildings[1].elec)))
f15.close()
print("Set 15 metric result written to file.")

print("All F1 scores written to file.")

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
f10 = open('C:/NILM/Data/Model_Train/Error_Output/Correct_Fraction/cor_frac10.txt','w')
f11 = open('C:/NILM/Data/Model_Train/Error_Output/Correct_Fraction/cor_frac11.txt','w')
f12 = open('C:/NILM/Data/Model_Train/Error_Output/Correct_Fraction/cor_frac12.txt','w')
f13 = open('C:/NILM/Data/Model_Train/Error_Output/Correct_Fraction/cor_frac13.txt','w')
f14 = open('C:/NILM/Data/Model_Train/Error_Output/Correct_Fraction/cor_frac14.txt','w')
f15 = open('C:/NILM/Data/Model_Train/Error_Output/Correct_Fraction/cor_frac15.txt','w')

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

print("Calucating for set 10")
f10.write(repr(fraction_energy_assigned_correctly(out10.buildings[1].elec, redd_data.buildings[1].elec)))
f10.close

print("Calucating for set 11")
f11.write(repr(fraction_energy_assigned_correctly(out11.buildings[1].elec, redd_data.buildings[1].elec)))
f11.close

print("Calucating for set 12")
f12.write(repr(fraction_energy_assigned_correctly(out12.buildings[1].elec, redd_data.buildings[1].elec)))
f12.close

print("Calucating for set 13")
f13.write(repr(fraction_energy_assigned_correctly(out13.buildings[1].elec, redd_data.buildings[1].elec)))
f13.close

print("Calucating for set 14")
f14.write(repr(fraction_energy_assigned_correctly(out14.buildings[1].elec, redd_data.buildings[1].elec)))
f14.close

print("Calucating for set 15")
f15.write(repr(fraction_energy_assigned_correctly(out15.buildings[1].elec, redd_data.buildings[1].elec)))
f15.close


print("Correct fraction calculations finished.")


#close datastores
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
out10.store.close()
out11.store.close()
out12.store.close()
out13.store.close()
out14.store.close()
out15.store.close()