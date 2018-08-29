from __future__ import division
import random 
import csv
import math
import pandas as pd
import fileinput
import sys

class Property:
	def __init__(self, batchCnt, sensorCnt, sampleCnt, propCnt):
		self.batchCnt=batchCnt
		self.sensorCnt=sensorCnt
		self.sampleCnt=sampleCnt
		self.propCnt=propCnt
	
	def loadData(self, inputFilePath):
		with fileinput.FileInput(inputFilePath, inplace=True) as file:
			for line in file:
				print(line.replace("\"", ""), end='')
		
		###printing file data
		print("\nFile data is:")
		file_list = []
		filereader = csv.reader(open(inputFilePath), delimiter =',')
		for row in filereader:
			#print(','.join(row))
			if len(row)>0:
				file_list.append(row)
		return file_list

	def netMean(self, file_list, batch_num, col_num, batchSize):
		#calculating the net mean
		net_sum=0
		net_mean = 0
		net_sum_list = []
		
		for outer_list in range((batchSize*batch_num+1), (batchSize*(batch_num+1)+1)):
			net_sum = net_sum + float(file_list[outer_list][col_num])
		net_mean = float(net_sum)/batchSize
		return net_mean

	def findMedian(self, file_list, batch_num, col_num, batchSize):
		print("batchSize =%d" %batchSize)
		mid_point = int(((batchSize*(batch_num+1)+1)-(batchSize*batch_num+1))/2)#calculating the median
		median = []
		for outer_list in range((batchSize*batch_num+1), (batchSize*(batch_num+1)+1)):
			#print("[%d] (%d)" %(outer_list, col_num))
			median.append(float(file_list[outer_list][col_num]))
		col_num +=1
		
		median.sort()
		if(batchSize%2==0):
			final_median = (float(median[mid_point-1]) + float(median[mid_point]))
			final_median = final_median/2
		else:
			final_median = float(median[mid_point])
		
		return final_median, median

	def findSTD(self, net_mean, median, batch_num, col_num, batchSize):
		#calculating the standard deviation
		std_dev = 0
		sum_diff = 0
		for n in range(batchSize):
			sum_diff = sum_diff + math.pow((median[n]-net_mean), 2)
		
		std_dev = math.sqrt(sum_diff/batchSize)
		return std_dev

	def findVariance(self, std_dev):
		#calculating the variance
		variance = 0
		variance = math.pow(std_dev, 2)
		return variance

	def findRange(self, median):
		#calculating the range
		range_data = 0
		range_data = max(median) - min(median)
		return range_data

	def findSkewness(self, net_mean, final_median, std_dev):
		#calculating the skewness
		skewness_coeff = 0
		skewness_coeff = float(3*(net_mean - final_median))/float(std_dev)
		return skewness_coeff

	def findKurtosis(self, net_mean, median, std_dev, batchSize):
		kurtosis = 0
		kurtosis_sum = 0
		
		for n in range(batchSize):
			kurtosis_sum = kurtosis_sum + (math.pow((median[n] - net_mean), 4))/batchSize
		
		kurtosis = kurtosis_sum / math.pow(std_dev, 4)
		return kurtosis

	def propExtraction(self, file_list, batch_num, col_num, batchSize):
		# Create a list to store calculated values
		final_calculated_values = []
		
		# Calculate Net-Mean values
		net_mean = self.netMean(file_list, batch_num, col_num, batchSize)
		final_calculated_values.append(net_mean)
		print("Net Mean value is: %f" %(net_mean))
		
		# Calculate Median values
		final_median, median = self.findMedian(file_list, batch_num, col_num, batchSize)
		final_calculated_values.append(final_median)
		print("Median value is: %f" %(final_median))
		
		# Calculate Maximum values
		final_calculated_values.append(max(list(map(float,median))))
		print("Max value is: %f" %(max(list(map(float,median)))))
		
		# Calculate Minimum values
		final_calculated_values.append(min(list(map(float,median))))
		print("Min value is: %f" %(min(list(map(float,median)))))
		
		# Calculate Standard Deviation values
		std_dev = self.findSTD(net_mean, median, batch_num, col_num, batchSize)
		final_calculated_values.append(std_dev)
		print("Standard deviation is: %f" %std_dev)   
		
		# Calculate Variance values
		variance = self.findVariance(std_dev)
		final_calculated_values.append(variance)
		print("Variance of data is: %f" %variance)
		
		range_data = self.findRange(median)
		final_calculated_values.append(range_data)
		print("Range of data is: %f" %range_data)
		
		# Calculate Skewness values
		skewness_coeff = self.findSkewness(net_mean, final_median, std_dev)
		final_calculated_values.append(skewness_coeff) 
		print("Skewness is: %f" %skewness_coeff)
		
		# Calculate kurtosis values
		kurtosis = self.findKurtosis(net_mean, median, std_dev, batchSize)
		final_calculated_values.append(kurtosis) 
		print("Kurtosis is: %f" %kurtosis)
		
		return final_calculated_values

def main():
	batchCnt=7
	sensorCnt=40
	sampleCnt=40
	propCnt=9
	
	inputFilePath = sys.argv[1] # "F:\study\data_preprocessed_python\data_preprocessed_python\\dataset.csv"
	outputFilePath = sys.argv[2] # "Output_data.csv"
	print("inputFilepath = %s" %inputFilePath)
	print("outputFIlePath = %s" %outputFilePath)
	
	prop = Property(batchCnt, sensorCnt, sampleCnt, propCnt)
	
	file_list = prop.loadData(inputFilePath)
	
	total_rows = len(file_list)
	total_cols = len(file_list[0])
	print(total_rows)
	batchSize = int(total_rows/(batchCnt*sampleCnt))
	
	#writing the final list to the file
	with open(outputFilePath,"w") as my_final_file:
		writer = csv.writer(my_final_file, lineterminator='\n')
		writer.writerow(["Net Mean", "Median", "Max", "Min", "STD", "Variance", "Range", "Skweness", "Kurtosis"])
		for batch_num in range(batchCnt*sampleCnt):
			for col_num in range(total_cols):
				print("Row from %d to %d" % (batchSize*batch_num+1, (batch_num+1)*batchSize))
				print("Col %d" % (col_num))
				final_calculated_values = prop.propExtraction(file_list, batch_num, col_num, batchSize)
				writer.writerow(final_calculated_values)
				final_calculated_values[:] = []

main()
