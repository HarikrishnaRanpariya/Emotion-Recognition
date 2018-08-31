# Import Library
from __future__ import print_function
from itertools import chain, product
import tensorflow as tf
import pandas as pd
import numpy as np
import fileinput
import random 
import math
import csv
import sys


def loadData(inputFilePath):
	"""Load data from the given 'inputFilePath' csv file
	
	Load csv file data and create a list
	
	Args:
		inputFilePath: .\prop\\prop_xx.csv
	
	Returns:
		file_list: List of input file data
	"""
	print("\nFile data is:")
	file_list = []
	# Load data from the input CSV file
	filereader = csv.reader(open(inputFilePath), delimiter =',')
	# Create a list by appending each row from the input file
	for row in filereader:
		if len(row)>0:
			file_list.append(row) # append row to list
	return file_list

def main():
	"""Convert data to Model data format
	
	Load inputfile data. Transform that loaded data into Model data 
	format and save them into outputfile.
	
	Args:
		sys.argv[1]: input file path (.\merge\merge_xx.csv)
		sys.argv[2]: output file path (.\modeldata\model_xx.csv)
	
	Returns:
		N/A
	
	INPUT: mergefile (.\merge\merge_xx.csv)
		size: (Trials*Property*Electrodes)x(Batches+Overall)
			  (  40  *   9    *   40     )x(   7   +   1   )
	OUTPUT: modelfile (.\modeldata\model_%s.csv)
		size: (Trials)x((Batches+Overall)*Property*Electrodes)
			  (  40  )x( (  7   +   1   )*   9    *    40    )
	"""
	batchCnt=7+1 # 7 Batches + 1 Overall property of their respective 7 Batches
	sensorCnt=40
	trialCnt=40
	propCnt=9
	
	# Get input file path from the command line
	inputFilePath = sys.argv[1]  # .\merge\merge_xx.csv
	print("inputFilepath = %s" %inputFilePath)
	
	# Get output file path from the command line
	outputFilePath = sys.argv[2] # .\modeldata\model_xx.csv
	print("outputFIlePath = %s" %outputFilePath)
	
	# Load data from input CSV file to file_list
	file_list = loadData(inputFilePath)
	
	total_rows = len(file_list)
	total_cols = len(file_list[0])
	
	# Calculate Batch size
	batchSize = int(total_rows/(batchCnt*trialCnt))
	
	# Open output file object
	with open(outputFilePath,"w") as my_final_file:
		# Get CSV file writer object
		writer = csv.writer(my_final_file, lineterminator='\n')
		title=[]
		
		# Create Title row of the output file
		# Loop for 'batchCnt' Batches  
		for batch in range(batchCnt):
			# Loop for 'sensorCnt' sensors of each batch
			for sensor in range(sensorCnt):
				# Loop for 'propCnt' properties of each sensor of each batch
				for prop in range(propCnt):
					title.append("b%ds%02dp%d" %(batch, sensor, prop))
		
		# Save title row to output file
		writer.writerow(title)
		
		# Batch wise Loop for whole dataset
		for batch_num in range(0,propCnt*trialCnt*sensorCnt, propCnt*sensorCnt):
			print("Row from %d to %d" % (batch_num, batch_num+propCnt*sensorCnt+1))
			
			# Transform whole batch data "(propCnt*sensorCnt) x batchCnt" into a raw data
			final_calculated_values = np.array(file_list[batch_num+1:batch_num+propCnt*sensorCnt+1]).transpose()
			
			# Save transformed row data to output file
			writer.writerow(list(chain.from_iterable(final_calculated_values)))

# Execution starts from here
main()

