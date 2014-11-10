from nilmtk.dataset import DataSet
from nilmtk.dataset_converters import convert_redd


#convert data from redd into a .hdf file for future loading
convert_redd("C:/NILM/Data_Sets/low_freq/","C:/Energy_Management/Data/redd_data.h5")

