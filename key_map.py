#Author: Michael Milicevich
#Class that maps appliance ID to datastore key for REDD Dataset

class Key_Map(object):

	'''
	This class contains a dict that maps all REDD DataStore keys for buildings
	1 to 6 to the name of the appliance (as provided by metadata). Class also outputs 
	appliance names to the user if needed.

	Parameters:
	------------

	building:  		The number (1-6) of the building instance you wish to map for
					for the dictionary keys.

	Attributes:
	------------

	map:			The dictionary that stores the key of an item, with the index
					being the REDD defined appliance name.

	'''
	def __init__(self, building = 1):
		self.map = {}
		self.load_map(building)


	def load_map(self, building):

		'''
		Function loads the appropiate dictionary according to the building
		instance passed (1-6). 

		Parameters:
		------------

		building:	The number (1-6) of the building instance you wish to map for
					for the dictionary keys.

		'''
		if building == 1:
			self.map['mains'] = '/building1/elec/meter1_2'
			self.map['mains1'] ='/building1/elec/meter1' 
			self.map['mains2'] ='/building1/elec/meter2' 
			self.map['fridge'] ='/building1/elec/meter5' 
			self.map['dish washer'] ='/building1/elec/meter6' 
			self.map['sockets1'] ='/building1/elec/meter7' 
			self.map['sockets2'] ='/building1/elec/meter8' 
			self.map['light1'] ='/building1/elec/meter9' 
			self.map['microwave'] ='/building1/elec/meter11' 
			self.map['unknown1'] ='/building1/elec/meter12' 
			self.map['electric space heater'] ='/building1/elec/meter13' 
			self.map['electric stove'] ='/building1/elec/meter14' 
			self.map['sockets3'] ='/building1/elec/meter15' 
			self.map['sockets4'] ='/building1/elec/meter16' 
			self.map['light2'] ='/building1/elec/meter17' 
			self.map['light3'] ='/building1/elec/meter18' 
			self.map['unknown2'] ='/building1/elec/meter19'
			self.map['washer dryer'] ='/building1/elec/meter10_20'
			self.map['electric oven'] ='/building1/elec/meter3_4'

		elif building == 2:
			self.map['mains'] = '/building2/elec/meter1_2'
			self.map['mains1'] = '/building2/elec/meter1'
			self.map['mains2'] = '/building2/elec/meter2'
			self.map['sockets1'] = '/building2/elec/meter3'
			self.map['light'] = '/building2/elec/meter4'
			self.map['electric stove'] = '/building2/elec/meter5'
			self.map['microwave'] = '/building2/elec/meter6'
			self.map['washer dryer'] = '/building2/elec/meter7'
			self.map['sockets2'] = '/building2/elec/meter8'
			self.map['fridge'] = '/building2/elec/meter9'
			self.map['dish washer'] = '/building2/elec/meter10'
			self.map['waste disposal unit'] = '/building2/elec/meter11'

		elif building == 3:
			self.map['mains'] = '/building3/elec/meter1_2'
			self.map['mains1'] = '/building3/elec/meter1'
			self.map['mains2'] = '/building3/elec/meter2'
			self.map['sockets1'] = '/building3/elec/meter3'
			self.map['sockets2'] = '/building3/elec/meter4'
			self.map['light1'] = '/building3/elec/meter5'
			self.map['CE Appliance'] = '/building3/elec/meter6'
			self.map['fridge'] = '/building3/elec/meter7'
			self.map['waste disposal unit'] = '/building3/elec/meter8'
			self.map['dish washer'] = '/building3/elec/meter9'
			self.map['electric furnace'] = '/building3/elec/meter10'
			self.map['light2'] = '/building3/elec/meter11'
			self.map['sockets3'] = '/building3/elec/meter12'
			self.map['washer dryer'] = '/building3/elec/meter13_14'
			self.map['light3'] = '/building3/elec/meter15'
			self.map['microwave'] = '/building3/elec/meter16'
			self.map['light4'] = '/building3/elec/meter17'
			self.map['smoke alarm'] = '/building3/elec/meter18'
			self.map['light5'] = '/building3/elec/meter19'
			self.map['unknown'] = '/building3/elec/meter20'
			self.map['sockets4'] = '/building3/elec/meter21'
			self.map['sockets5'] = '/building3/elec/meter22'

		elif building == 4:
			self.map['mains'] = '/building4/elec/meter1_2'
			self.map['mains1'] = '/building4/elec/meter1'
			self.map['mains2'] = '/building4/elec/meter2'
			self.map['light1'] = '/building4/elec/meter3'
			self.map['electric furnace'] = '/building4/elec/meter4'
			self.map['sockets1'] = '/building4/elec/meter5'
			self.map['sockets2'] = '/building4/elec/meter6'
			self.map['washer dryer'] = '/building4/elec/meter7'
			self.map['electric stove'] = '/building4/elec/meter8'
			self.map['air conditioner1'] = '/building4/elec/meter9_10'
			self.map['unknown1'] = '/building4/elec/meter11'
			self.map['smoke alarm'] = '/building4/elec/meter12'
			self.map['light2'] = '/building4/elec/meter13'
			self.map['sockets3'] = '/building4/elec/meter14'
			self.map['dish washer'] = '/building4/elec/meter15'
			self.map['unknown2'] = '/building4/elec/meter16'
			self.map['unknown3'] = '/building4/elec/meter17'
			self.map['light3'] = '/building4/elec/meter18'
			self.map['light4'] = '/building4/elec/meter19'
			self.map['air conditioner2'] = '/building4/elec/meter20'

		elif building == 5:
			self.map['mains'] = '/building5/elec/meter1_2'
			self.map['mains1'] = '/building5/elec/meter1'
			self.map['mains2'] = '/building5/elec/meter2'
			self.map['microwave'] = '/building5/elec/meter3'
			self.map['light1'] = '/building5/elec/meter4'
			self.map['sockets1'] = '/building5/elec/meter5'
			self.map['electric furnace'] = '/building5/elec/meter6'
			self.map['sockets2'] = '/building5/elec/meter7'
			self.map['subpanel1'] = '/building5/elec/meter10'
			self.map['subpanel2'] = '/building5/elec/meter11'
			self.map['light2'] = '/building5/elec/meter14'
			self.map['sockets3'] = '/building5/elec/meter15'
			self.map['unknown'] = '/building5/elec/meter16'
			self.map['light3'] = '/building5/elec/meter17'
			self.map['fridge'] = '/building5/elec/meter18'
			self.map['light4'] = '/building5/elec/meter19'
			self.map['dish washer'] = '/building5/elec/meter20'
			self.map['waste disposal unit'] = '/building5/elec/meter21'
			self.map['CE appliance'] = '/building5/elec/meter22'
			self.map['light5'] = '/building5/elec/meter23'
			self.map['sockets4'] = '/building5/elec/meter24'
			self.map['sockets5'] = '/building5/elec/meter25'
			self.map['sockets6'] = '/building5/elec/meter26'
			self.map['electric space heater'] = '/building5/elec/meter12_13'
			self.map['washer dryer'] = '/building5/elec/meter8_9'

		elif building == 6:
			self.map['mains'] = '/building6/elec/meter1_2'
			self.map['mains1'] = '/building6/elec/meter1'
			self.map['mains2'] = '/building6/elec/meter2'
			self.map['sockets1'] = '/building6/elec/meter3'
			self.map['washer dryer'] = '/building6/elec/meter4'
			self.map['electric stove'] = '/building6/elec/meter5'
			self.map['CE appliance'] = '/building6/elec/meter6'
			self.map['unknown1'] = '/building6/elec/meter7'
			self.map['fridge'] = '/building6/elec/meter8'
			self.map['dish washer'] = '/building6/elec/meter9'
			self.map['sockets2'] = '/building6/elec/meter10'
			self.map['sockets3'] = '/building6/elec/meter11'
			self.map['electric space heater'] = '/building6/elec/meter12'
			self.map['sockets4'] = '/building6/elec/meter13'
			self.map['light'] = '/building6/elec/meter14'
			self.map['air handling unit'] = '/building6/elec/meter15'
			self.map['air conditioner'] = '/building6/elec/meter16_17'

		else:
			print("Error: Building application key map cannot be found.")


	def list_appliances(self):
		'''
		Function prints out all of the different appliance names found in 
		the current object instance.

		'''
		for key in self.map:
			print(key)

	def get_key(self, apl):
		'''
		Function returns the DataStore key based on appliance
		name input.

		Parameters:
		------------

		apl:			The name of the appliance (as defined in metadata)

		'''
		return self.map[apl]

