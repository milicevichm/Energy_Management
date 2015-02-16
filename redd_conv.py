from nilmtk.dataset import DataSet
from nilmtk.dataset_converters import convert_redd


#convert data from redd into a .hdf file for future loading
convert_redd("C:/NILM/Data/REDD/low_freq/","C:/NILM/Data/Output/redd.h5")

