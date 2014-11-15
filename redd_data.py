#Author: Michael Milicevich
#Object allows for quick plotting and testing of NILMTK Data

#import modules
from __future__ import print_function, division
from nilmtk import HDFDataStore, DataSet
from nilmtk.disaggregate import CombinatorialOptimisation
import pandas as pd
import matplotlib.pyplot as plt
from key_map import *
import warnings
from nilmtk.plots import plot_series


#supress warnings to users console
warnings.filterwarnings("ignore")


class REDD_Data(object):

	'''
	REDD_Data Class is an object designed to abstract the lower level commands of
	the NILMTK software package, with focus on the use of REDD DataSet. Function is 
	designed to allow rapid experimentation and disaggregation compared to attempting 
	to set package up from scratch.

	This class requires the following for proper usage:
	- NILMTK package: https://github.com/nilmtk
	- REDD Dataset (converted to .h5): redd.csail.mit.edu
	- Various dependancies (that NILMTK also requires), most can be downloaded through
	  Anaconda: continuum.io/downloads


	Parameters
	-----------
	in_filepath:		Filepath of converted REDD dataset (in .h5 format)
	out_filepath:		filepath to place output disaggregation dataset (in .h5 format)

	Attributes
	-----------
	km: Key_Map Object
		initializes the key_map object which will allow for the mapping of a meters
		appliance name to its specific .H5 key.

	dataStore: NILMTK HDFDataStore Object
		the HDFDataStore that will contain the converted REDD DataSet.

	dataSet: NILMTK DataSet Object
		the DataSet object that is generated from the REDD DataStore (self.dataStore)		

	outDataStore: NILMTK HDFDataStore Object
		the HDFDataStore that will contain the disaggregated dataset.

	co: NILMTK CombinatorialOptimisation object
		the disaggregation model object that will be trained and will disaggregate the 
		working dataset

	train_group: NILMTK MeterGroup object
		the MeterGroup object that is used to train the disaggregation model (self.co)

	'''
	def __init__ (self,in_filepath,out_filepath):
		print("Loading DataStore and Generating Dataset...")
		self.km = {}
		self.dataStore = HDFDataStore(in_filepath)
		self.dataSet = DataSet()
		self.dataSet.load(self.dataStore)
		self.outDataStore = HDFDataStore(out_filepath,'w')
		self.co = CombinatorialOptimisation()
		self.train_group = {}
		print("Data Properly Loaded!")


	def train_disag_model(self,building_inst, use_topk = False, k = 5):
		'''
		Function trains the disaggregation model using a selected MeterGroup.

		Parameters
		-----------

		building_inst: 	the instance # of the building that you wish to grab the 
					   	training group from.

		use_topk:		true if you wish to only grab the top k most energy intensive
						appliance to train the model, false if you wish to use all
						appliances.

		k:				the # of appliances you wish to use (if use_topk = True)

		'''

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
		'''
		Function loads the disaggregation model to a file.

		Parameters
		-----------

		filepath:	exact filepath of the model file.

		'''
		print("Loading CO Disaggreation Model...")
		self.co.import_model(filepath)
		print("Model Sucessfully Loaded!")
		

	def save_disag_model(self,filepath):
		'''
		Function saves the disaggregation model to a file.

		Parameters
		-----------

		filepath:	exact filepath of the model file.

		'''
		print("Saving CO Disaggregation Model...")
		self.co.export_model(filepath)
		print("Model Sucessfully Saved!")


	def disaggregate(self,building_inst):
		'''
		Function will disaggregate the mains MeterGroup of the passed building 
		instance, and save this to the self.outDataStore object.

		Parameters
		-----------

		building_inst:	instance # of the building mains you wish to disaggregate.

		'''
		print("Disaggregating Building Mains...")		
		self.co.disaggregate(self.dataSet.buildings[building_inst].elec.mains(),self.outDataStore)
		print("Mains sucessfully disaggregated!")


	def close(self):
		'''
		Function closes all open DataStore's being used by the program.

		'''
		print("Closing DataStores...")
		self.dataStore.close()
		self.outDataStore.close()
		print("Output DataStores Sucessfully Closed")
		

	'''
	All Plot Functions below are a WORK IN PROGRESS!-----------------------------------
	Documentation will be provided upon completion.------------------------------------

	'''
		

	def plot_disag_apl(self,inst,appliance,t1="",t2=""):
		self.km = Key_Map(inst)
		plot_series(self.outDataStore.store.get(self.km.get_key(appliance))[t1: t2])
		plt.title("Disaggregated " + appliance.capitalize()+" Energy") 
		plt.show()

	
	def show_plots(self):
		plt.show()


	def building_plot_all(self,building_inst,t1,t2):
		self.dataSet.buildings[building_inst].elec.plot(t1,t2)
		plt.title("Building "+str(building_inst)+" Energy per Appliance")
		plt.ylabel('Power [W]')
		plt.xlabel('Hour')


	def plot_redd_mains_data(self, inst=1, t1 = "", t2 = ""):
		self.km = Key_Map(inst)
		series1 = self.dataStore.store.get(self.km.get_key('mains1'))[t1:t2]
		series2 = self.dataStore.store.get(self.km.get_key('mains2'))[t1:t2]
		plot_series(series1 + series2)
		plt.title("Building "+str(inst)+" Mains Energy")
		plt.show()



		
