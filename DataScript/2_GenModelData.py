# Import Library
from __future__ import division
from os import walk
import os 
import sys
import errno

def main():
	"""Convert Raw 'data' format to Model data format
	
	This function extracts batchwise N features from each user trials and 
	convert them into the Model data format. It executes 3 separate 
	scripts to extract properties from the each batch and overall 
	properties of each trials. It also generate two intermediate 
	files (.\property\prop_xx.csv & .\merge\merge_xx.csv) and one output 
	file (.\modeldata\model_%s.csv). 
	
	Args:
		sys.argv[1]: Raw 'data' folder path (.\data\\) which generated 
		using ExtractRawData.py script.
	
	Returns:
		N/A
	
	Property: Total 9 important properties we are calculating from the 
			  given dataset to reduce overall data size.
			  
			 [Net_Mean		Median		Max		
			  Min			STD			Variance
			  Range			Skewness	Kurtosis]

	SubScripts:
		Batch_Prop.py: 
			INPUT: inputfilePath+datafile (.\data\data_xx.csv)
				size: (Trials*Samples)x(Electrodes)
					  (  40  * 8064  )x(    40   )
			OUTPUT: propfile (.\property\prop_xx.csv)
				size: (Trials*Electrodes*Batches)x(Property)
					  (  40  *    40    *   7   )x(    9   )
		OverAll_Prop.py 
			INPUT: propfile (.\property\prop_xx.csv)
				size: (Trials*Electrodes*Batches)x(Property)
					  (  40  *    40    *   7   )x(    9   )
			OUTPUT: mergefile (.\merge\merge_xx.csv)
				size: (Trials*Property*Electrodes)x(Batches+Overall)
					  (  40  *   9    *   40     )x(   7   +   1   )
		ConvToModelData.py 
			INPUT: mergefile (.\merge\merge_xx.csv)
				size: (Trials*Property*Electrodes)x(Batches+Overall)
					  (  40  *   9    *   40     )x(   7   +   1   )
			OUTPUT: modelfile (.\modeldata\model_%s.csv)
				size: (Trials)x((Batches+Overall)*Property*Electrodes)
					  (  40  )x( (  7   +   1   )*   9    *    40    )
		
	"""
	# Save raw 'data' folder path
	inputfilePath = sys.argv[1]
	f = []
	
	# Read all filenames with .csv extention in the given folder
	for (dirpath, dirnames, filenames) in walk(inputfilePath):
		filenames = [i for i in filenames if i.endswith('.csv')] 
		f.extend(filenames)
		break
	
	# Loop for each 'data' csv files
	for datafile in f:
		# Create output file path for Batch_Prop.py script 
		propfile = (".\property\prop_%s.csv" %(datafile.split('.')[0][5:7]))
		# Check given data file path exists or not.
		# If it does not then create new directory path using given name
		if not os.path.exists(os.path.dirname(propfile)):
			try:
				os.makedirs(os.path.dirname(propfile)) # Create new Dir
			except OSError as exc: # Guard against race condition
				if exc.errno != errno.EEXIST:
					raise
		print(propfile)
		
		# Create output file path for OverAll_Prop.py script 
		mergefile = (".\merge\merge_%s.csv" %(datafile.split('.')[0][5:7]))
		# Check given data file path exists or not.
		# If it does not then create new directory path using given name
		if not os.path.exists(os.path.dirname(mergefile)):
			try:
				os.makedirs(os.path.dirname(mergefile)) # Create new Dir
			except OSError as exc: # Guard against race condition
				if exc.errno != errno.EEXIST:
					raise
		print(mergefile)
		
		# Create output file path for ConvToModelData.py script 	
		modelfile = (".\modeldata\model_%s.csv" %(datafile.split('.')[0][5:7]))
		# Check given data file path exists or not.
		# If it does not then create new directory path using given name
		if not os.path.exists(os.path.dirname(modelfile)):
			try:
				os.makedirs(os.path.dirname(modelfile)) # Create new Dir
			except OSError as exc: # Guard against race condition
				if exc.errno != errno.EEXIST:
					raise
		print(modelfile)
		
		# Execute Batch_Prop.py script to calculate batchwise properties
		os.system('python Batch_Prop.py %s %s' %(inputfilePath+datafile, propfile))
		print("Properties are calculated: from %s to %s " %(inputfilePath+datafile, propfile))
		
		# Execute OverAll_Prop.py script to calculate overall properties of each trials from their batchwise data 
		os.system('python OverAll_Prop.py %s %s' %(propfile, mergefile))
		print("Property Merger is done: from %s to %s " %(propfile, mergefile))
		
		# Execute ConvToModelData.py script to convert all trials' property data to Model data format
		os.system('python ConvToModelData.py %s %s' %(mergefile, modelfile))
		print("Coverted to Model data: from %s to %s " %(mergefile, modelfile))

# Execution starts from here
main()

