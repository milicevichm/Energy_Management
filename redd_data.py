#Author: Michael Milicevich
#Object allows for quick plotting and testing of NILMTK Data

from __future__ import print_function, division
from nilmtk import HDFDataStore, DataSet
from nilmtk.disaggregate import CombinatorialOptimisation
import pandas as pd
import matplotlib.pyplot as plt
from key_map import *
import warnings
from nilmtk.utils import *
from nilmtk.plots import *


#supress warnings
warnings.filterwarnings("ignore")

class REDD_Data(object):

	def __init__ (self,in_filepath,out_filepath):
		print("Loading DataStore and Generating Dataset...")
		self.dataStore = HDFDataStore(in_filepath)
		self.dataSet = DataSet()
		self.dataSet.load(self.dataStore)
		self.outDataStore = HDFDataStore(out_filepath,'w')
		self.co = CombinatorialOptimisation()
		print("Data Properly Loaded!")


	def train_disag_model(self,building_inst, use_topk = False, k = 5):
		print("Training CO Disaggregation Model using given metergroup...")

		if (building_inst <= 6) & (building_inst > 0): 
			#Select appropiate meter group to train with
			if use_topk == True:
				self.train_group = self.dataSet.buildings[building_inst].elec.select_top_k(k)
			else:
				self.train_group = self.dataSet.buildings[building_inst].elec

			self.co.train(self.train_group)
			print("CO Disaggreation Model Sucessfully Trained!")

		else:
			print("Error: Please select a building_inst of 1-6.")
			print("Model unsucessfully trained.")

	def load_disag_model(self, filepath):
		print("Loading CO Disaggreation Model...")
		self.co.import_model(filepath)
		print("Model Sucessfully Loaded!")
		

	def save_disag_model(self,filepath):
		print("Saving CO Disaggregation Model...")
		self.co.export_model(filepath)
		print("Model Sucessfully Saved!")


	def disaggregate(self,building_inst):
		print("Disaggregating Building Mains...")		
		self.co.disaggregate(self.dataSet.buildings[building_inst].elec.mains(),self.outDataStore)
		print("Mains sucessfully disaggregated!")

	def close(self):
		print("Closing DataStores...")
		self.dataStore.close()
		self.outDataStore.close()
		print("Output DataStores Sucessfully Closed")


	def plot_mains(self,building_inst, t_start, t_end, mains_inst = 1):
		self.km = Key_Map(building_inst)
		#plot mains data vs disaggregated data based on appliance key
		if mains_inst == 1:
			self.dataStore.store.get(self.km.get_key('mains1'))[t_start : t_end].plot()
		else:
			self.dataStore.store.get(self.km.get_key('mains2'))[t_start : t_end].plot()

		plt.title("Aggregated Mains Energy") 
		plt.legend().set_visible(False)
		plt.ylabel('Apparent Power [VA]')
		plt.xlabel('Hour')
		

	def plot_disag_apl(self,building_inst,appliance_name,t_start,t_end):
		self.km = Key_Map(building_inst)
		self.apl_key = self.km.get_key(appliance_name)
		self.outDataStore.store.get(self.apl_key)[t_start : t_end].plot()
		plt.title("Disaggregated " + appliance_name.capitalize()+" Energy") 
		plt.legend().set_visible(False)
		plt.ylabel('Apparent Power [VA]')
		plt.xlabel('Hour')
	
	def show_plots(self):
		plt.show()

	def building_plot_all(self,building_inst,t1,t2):
		self.dataSet.buildings[building_inst].elec.plot(t1,t2)
		plt.title("Building "+str(building_inst)+" Energy per Appliance")
		plt.ylabel('Apparent Power [VA]')
		plt.xlabel('Hour')

	def plot_redd_mains_data(self, inst=1, t1 = "", t2 = ""):
		self.km = Key_Map(inst)
		#TODO: Look into appending both datastores together to build one mains set
		plot_series(self.dataStore.store.get(self.km.get_key('mains1'))[t1:t2])
		plot_series(self.dataStore.store.get(self.km.get_key('mains2'))[t1:t2])



		
