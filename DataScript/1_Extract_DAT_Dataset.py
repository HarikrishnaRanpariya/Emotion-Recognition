# NOTE: This file executes another file which uses cPickle to load 
#		data. Only Python 2.7 or older version supports cPickle. So for 
#		successful execution of this file Python 2.7 or older version is
# 		required

# Import Library
from __future__ import division
from os import walk
import os 
import sys

def main():
	"""Extract data from the DEAP Dataset.
	
	Extract raw EEG data lists from each user's experiment files. There
	are total 32 user experiment files (.dat). Each file has two lists: 
	'data' list and its corresponding 'label' list. we are saving both 
	these data-lists in seperate folder & files 'data' list stores as 
	'.\data\data_xx.csv' file and 'label' list stores as 
	'.\label\label_xx.csv' file.
	
	Args:
		sys.argv[1]: DEAP dataset folder path
		
	Returns:
		N/A
	"""
	
	inputfilePath = sys.argv[1] # Save the raw dataset folder path
	f = []
	
	# Read all '.dat' extension file names from the given folder path
	for (dirpath, dirnames, filenames) in walk(inputfilePath):
		filenames = [i for i in filenames if i.endswith('.dat')] 
		f.extend(filenames)
		break
	
	# Extract 'data' & 'label' lists from each '.dat' files and save 
	# them as '.csv' files in their respective folders
	for fileName in f:
		
		inputFile = inputfilePath+fileName
		
		datafile = (".\data\data_%s.csv" %(fileName.split('.')[0][1:3]))
		labelfile = (".\label\label_%s.csv" %(fileName.split('.')[0][1:3]))
		
		# Execute '_1_DATToCSV.py' file to extract arrays from 
		# '.dat' files
		os.system('python DATToCSV.py %s' %inputFile)
		print("Converted raw data from %s to %s & %s" %(fileName, datafile, labelfile))

# Execution starts from here
main()

