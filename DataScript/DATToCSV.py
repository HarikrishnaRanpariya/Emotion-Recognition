# NOTE: This file uses cPickle to load data. Only Python 2.7 or older 
# 		versions supports cPickle. So this file will only be 
# 		successfully executed with Python 2.7 or older versions.

# Import library
import cPickle 
import numpy as np
import csv
import sys
import os
import errno

# Declare Global Constants
TRIAL = 40
ELEC = 40
SAMPLE = 8064
LABEL = 4
DEBUG = False
SENSORCNT=40

# Define Global Macros
dat = [TRIAL * SAMPLE *[ELEC*[0]]]
label =  [TRIAL *[LABEL*[0]]]

def log(s):
	"""Debug LOG fucntion
	
	Args:
		s: Log string
	
	Returns:
		N/A
	"""
	
	# Check DEBUG macro is enable before printing logs
    if DEBUG:
        print(s) 

def loadData(FileName):
	"""Load data from the given file
	
	Using cPickle function it loads two lists ('data' & 'label') from 
	the given the 'FileName'. It loads both list-pointers in the 'dataset' 
	variable.
	
	Args:
		FileName: Raw datafile path '.dat'
	
	Returns:
		dataset: datarray pointers:
					 dataset[0] is pointing to 'data' list
					 dataset[1] is pointing to 'label' list
	"""
	# Load both data pointers using cPickle (only executes in PY 2.7)
	dataset = cPickle.load(open(FileName, 'rb'))
	
	return dataset

def extractArray(dataFilePath, dataType, dataset):
	"""Save a dataset list in given file.
	
	This function extract specific(dataType: 'data' or 'label') list 
	from the given 'dataset' pointer and save as it is as it extracted 
	from the '.dat' file in 'dataFilePath' file path.
	
	Args:
		dataFilePath: It has data filepath where user wants to store 
		extracted list.
		dataType: It contains either 'data' or 'label' value to indicate 
		for which list user wants to extract from given dataset pointer.
		dataset: It is a dataset pointer from data list will be extract
	
	Returns:
		N/A
		INPUT Data: It contains two lists
			'data': experiment raw EEG data of each user
				size: (Trials*Samples)x(Electrodes)
					  (  40  * 8064  )x(    40   )
			'label': experiment label data of each user
				size: (Trials)x(Labels)
					  (  40  )x(  4   )
		OUTPUT Data: It will have two separate files
			'.\data\data_xx.csv':
				size: Same as 'data' list in INPUT file
			'.\label\label_xx.csv':
				size: Same as 'label' list in INPUT file
				 
	"""
	
	# Check given data file path exists or not.
	# If it does not then create new directory path using given name
	if not os.path.exists(os.path.dirname(dataFilePath)):
		try:
			os.makedirs(os.path.dirname(dataFilePath)) # Create new Dir
		except OSError as exc: # Guard against race condition
			if exc.errno != errno.EEXIST:
				raise
	
	# Open 'dataFilePath' file 
	with open(dataFilePath, 'wb') as datafile:
		
		# Get output csv file pointer to store data in csv format
		FData = csv.writer(datafile, quoting=csv.QUOTE_ALL, lineterminator='\n')
		title=[]
		
		# Append Title raw in output file based on their dataType
		if dataType is "data":
			for sensor in range(SENSORCNT):
				title.append("S%02d" %(sensor))
			FData.writerow(title)
		else:
			title=["Valance", "Arousal", "Dominance", "Liking"]
			FData.writerow(title)
		
		# Loop for 'TRIAL' trials
		for trial in range(0, TRIAL):			
			
			if dataType is "data":
				# Loop for 'SAMPLE' Samples of each trial
				for sample in range(0, SAMPLE):
					# Loop for 'ELEC' electrodes of each sample of each trial 
					for elec in range(0, ELEC):
						dat[0][trial*SAMPLE+sample][elec] = dataset['data'][trial][elec][sample]
						log("dat[%d][%d] = %d "%(trial*SAMPLE+sample, elec, dat[0][trial*SAMPLE+sample][elec]))
					log(dat[0][trial*SAMPLE+sample][2])
					# Save each row of data in the output CSV file
					FData.writerow(dat[0][trial*SAMPLE+sample])
			else:
				# Loop for 'LABEL' labels of their respective trials
				for label_ in range(0, LABEL):
					label[0][trial][label_] = dataset['labels'][trial][label_]
					log("label[%d][%d] = %d "%(trial, label_, label[0][trial][label_]))
				# Save each row of label data in the output CSV file
				FData.writerow(label[0][trial])
			log("\n")
	# Close output file object
	datafile.close()

def main():
	"""Save a dataset list in given file.
	
	This main function will call  functions to extract both 'data' & 
	'label' lists from the given input file and store them in their 
	respective filepath.
	
	Args:
		sys.argv[1]: Input raw data file path(.dat)
	
	Returns:
		N/A
	"""
	# Load raw data list pointers using cPickle
	log("Load data of %s" %(sys.argv[1]))
	dataset = loadData(sys.argv[1])
	
	# Save the 'data' list from the given data pointer.
	log("extract data array from %s" %(sys.argv[1]))
	extractArray(".\data\data_%02d.csv" %(int(filter(str.isdigit, sys.argv[1]))), "data", dataset)
	
	# Save the 'label' list from the given data pointer.
	log("extract label array from %s" %(sys.argv[1]))
	extractArray(".\label\label_%02d.csv" %(int(filter(str.isdigit, sys.argv[1]))), "label", dataset)

# Execution starts from here
main()
