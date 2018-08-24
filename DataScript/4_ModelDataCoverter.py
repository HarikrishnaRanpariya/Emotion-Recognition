from __future__ import print_function

#importing data
import numpy as np
import tensorflow as tf
import pandas as pd
import random 
import csv
import math
import fileinput
import sys
from itertools import chain, product

def loadData(inputFilePath):
	print("\nFile data is:")
	file_list = []
	filereader = csv.reader(open(inputFilePath), delimiter =',')
	for row in filereader:
		if len(row)>0:
			file_list.append(row)
	return file_list

def main():
	batchCnt=7+1
	sensorCnt=40
	sampleCnt=40
	propCnt=9
	print("hello")
	inputFilePath = sys.argv[1] # "F:\study\data_preprocessed_python\data_preprocessed_python\\dataset.csv"
	outputFilePath = sys.argv[2] # "Output_data.csv"
	print("inputFilepath = %s" %inputFilePath)
	print("outputFIlePath = %s" %outputFilePath)
	
	file_list = loadData(inputFilePath)
	
	total_rows = len(file_list)
	total_cols = len(file_list[0])
	
	print(total_rows)
	batchSize = int(total_rows/(batchCnt*sampleCnt))
	#writing the final list to the file
	with open(outputFilePath,"w") as my_final_file:
		writer = csv.writer(my_final_file, lineterminator='\n')
		title=[]
		for batch in range(batchCnt):
			for sensor in range(sensorCnt):
				for prop in range(propCnt):
					title.append("b%ds%02dp%d" %(batch, sensor, prop))
		#writer.writerow(["Net Mean", "Median", "Max", "Min", "STD", "Variance", "Range", "Skweness", "Kurtosis"])
		writer.writerow(title)
		
		for batch_num in range(0,propCnt*sampleCnt*sensorCnt, propCnt*sensorCnt):
			print("Row from %d to %d" % (batch_num, batch_num+propCnt*sensorCnt+1))
			#range(0,batchCnt*sampleCnt*sensorCnt, batchCnt*sensorCnt)  #range(batchCnt*sampleCnt):
			final_calculated_values = np.array(file_list[batch_num+1:batch_num+propCnt*sensorCnt+1]).transpose()
			writer.writerow(list(chain.from_iterable(final_calculated_values)))
			#final_calculated_values[:] = []

main()

