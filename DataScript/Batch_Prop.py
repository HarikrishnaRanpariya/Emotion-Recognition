
# import libraries
from __future__ import division
import random 
import csv
import math
import pandas as pd
import fileinput
import sys

class Property:
	"""This class extracts the data properties.
	
	The property class is responsible for the extraction of different 
	properties of data such as mean,median,standard deviation etc.
	
	Arguments:
	batchCnt : It is the total number of batches into which we are dividing 
	each electrode's reading values.
	sensorCnt : total number of sensor being used.
	trialCnt : this represents the total number of trials or videos
	propCnt : this represents the number of properties we are extracting
	
	"""
	def __init__(self, batchCnt, sensorCnt, trialCnt, propCnt):
		# Inits Property class with batchCnt,sensorCnt,trialCnt,propCnt.
		
		self.batchCnt=batchCnt
		self.sensorCnt=sensorCnt
		self.trialCnt=trialCnt
		self.propCnt=propCnt
	
	def loadData(self, inputFilePath):
		"""this method loads data from data folder
		
		It takes the address of the data folder and loads the data files 
		from the data folder.
		
		Args:
		inputFilePath : takes the address of the data folder as argument
		
		"""
		# collect all files in the data folder as File
		with fileinput.FileInput(inputFilePath, inplace=True) as File:
			# loop over File variable
			for line in File:
				# convert the strings data in file into integer
				print(line.replace("\"", ""), end='')
		
		# print the file data
		print("\nFile data is:")
		file_list = []
		# open the data file and read it's content
		filereader = csv.reader(open(inputFilePath), delimiter =',')
		# loop over the filereader data row-wise
		for row in filereader:
			# check the length of row
			if len(row)>0:
				# append to file_list
				file_list.append(row)
		return file_list

	def netMean(self, file_list, batch_num, col_num, batchSize):
		"""Calculates the net mean
		
		This method calculates net mean of data using certain arguments
		like file_list,batch_num,col_num, batchsize
		
		Args:
		file_list : It contains the data rows as it's each element.
		batch_num : It's product of batchCnt and trialCnt.
		col_num   : total number of columns in data
		batchSize : (total_rows/(batchCnt*trialCnt))
		
		Returns:
		It returns net calculated mean as the final output
		
		"""
		net_sum=0
		net_mean = 0
		
		# iterate over file_list to calculate net_sum
		for outer_list in range((batchSize*batch_num+1), (batchSize*(batch_num+1)+1)):
			net_sum = net_sum + float(file_list[outer_list][col_num])
		# divide the net_sum by batchSize to get the final net_mean
		net_mean = float(net_sum)/batchSize
		return net_mean

	def findMedian(self, file_list, batch_num, col_num, batchSize):
		"""Calculates the median
		
		This method takes file_list, batch_num and col_num along with
		batchsize as arguments to calculate the median
		
		Args:
		file_list : It contains the data rows as it's each element.
		batch_num : It's product of batchCnt and trialCnt.
		col_num   : total number of columns in data
		batchSize : (total_rows/(batchCnt*trialCnt))
		
		Returns:
		It returns final median as the output.
		
		"""
		print("batchSize =%d" %batchSize)
		
		# Calculating the median
		mid_point = int(((batchSize*(batch_num+1)+1)-(batchSize*batch_num+1))/2)
		
		# Creating an empty list to store the median values
		median = []
		
		# iterate over file_list to calculate net_sum
		for outer_list in range((batchSize*batch_num+1), (batchSize*(batch_num+1)+1)):
			median.append(float(file_list[outer_list][col_num]))
		col_num +=1
		
		# sort the median list to calculate the final median value
		median.sort()
		
		# calculate the median based on the odd or even batchSize
		if(batchSize%2==0):
			final_median = (float(median[mid_point-1]) + float(median[mid_point]))
			final_median = final_median/2
		else:
			final_median = float(median[mid_point])
		
		#return median values
		return final_median, median

	def findSTD(self, net_mean, median, batch_num, col_num, batchSize):
		"""Calculates the standard deviation
		
		This method caculates the standard devaiation values using net_mean,
		median,batch_num,col_num,batchSize
		
		Args:
		net_mean : contains the net_mean value from the mean function
		median : uses the median value caculated in the last function
		batch_num : It's product of batchCnt and trialCnt.
		col_num  : total number of columns in data
		batchSize : (total_rows/(batchCnt*trialCnt))
		
		Returns:
		It returns the calculated standard deviation value.
		
		"""
		
		# initialize the variables
		std_dev = 0
		sum_diff = 0
		
		# iterate over batch to calculate the sum_diff value
		for n in range(batchSize):
			sum_diff = sum_diff + math.pow((median[n]-net_mean), 2)
		
		# caculate the standard deviation 
		std_dev = math.sqrt(sum_diff/batchSize)
		
		# return the final value
		return std_dev

	def findVariance(self, std_dev):
		"""Calculates the variance
		
		This method takes the standard deviation as input and finds the
		variance value.
		
		Args:
		std_dev : Standard deviation value calculated in the last function
		
		Returns:
		Variance value
		
		"""
		
		#Calculating the variance
		variance = 0
		variance = math.pow(std_dev, 2)
		return variance

	def findRange(self, median):
		"""Calculates the range
		
		Takes median as argument and calculates the range value.
		
		Args:
		median : median value is the only input
		
		Returns:
		Returns the range value
		"""
		range_data = 0
		
		# Calculates the range
		range_data = max(median) - min(median)
		return range_data

	def findSkewness(self, net_mean, final_median, std_dev):
		"""Calculates the skewness
		
		Takes final_median,net_mean and standard deviation as input and
		calculates the skewness value.
		
		Args:
		net_mean : Net mean value
		final_median : Final median value
		std_dev : the standard deviation value
		"""
		skewness_coeff = 0
		
		# Calculating skewness
		skewness_coeff = float(3*(net_mean - final_median))/float(std_dev)
		return skewness_coeff

	def findKurtosis(self, net_mean, median, std_dev, batchSize):
		"""Caculates Kurtosis value
		
		Args:
		net_mean : Net mean value
		median : median value
		std_dev : Standard Deviation value
		batchSize : (total_rows/(batchCnt*trialCnt))
		
		Returns :
		It returns the final Kurtosis value as the end result.
		
		"""
		
		kurtosis = 0
		kurtosis_sum = 0
		
		# Iterate over batches and get the kurtosis sum.
		for n in range(batchSize):
			kurtosis_sum = kurtosis_sum + (math.pow((median[n] - net_mean), 4))/batchSize
		
		# calculate the kurtosis value
		kurtosis = kurtosis_sum / math.pow(std_dev, 4)
		return kurtosis

	def propExtraction(self, file_list, batch_num, col_num, batchSize):
		"""Calls the respective property methods 
		
		This method calls the repective property methods and append the 
		values to the final_calculated_values list.
		
		Args:
		file_list : It contains the data rows as it's each element.
		batch_num : It's product of batchCnt and trialCnt.
		col_num   : total number of columns in data
		batchSize : (total_rows/(batchCnt*trialCnt))
		
		Returns :
		List containing all the final property values.
		"""
		
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
	trialCnt=40
	propCnt=9
	
	# Pass the dataset folder path as argument in command line
	inputFilePath = sys.argv[1] 
	# Pass the name of the output file
	outputFilePath = sys.argv[2]
	print("inputFilepath = %s" %inputFilePath)
	print("outputFIlePath = %s" %outputFilePath)
	
	# create an object of Property class and pass the arguments
	prop = Property(batchCnt, sensorCnt, trialCnt, propCnt)
	
	# call the loadData method and pass the data file
	file_list = prop.loadData(inputFilePath)
	
	# get the total_rows, total_cols, batchSize values
	total_rows = len(file_list)
	total_cols = len(file_list[0])
	print(total_rows)
	batchSize = int(total_rows/(batchCnt*trialCnt))
	
	#writing the final list to the file
	with open(outputFilePath,"w") as my_final_file:
		writer = csv.writer(my_final_file, lineterminator='\n')
		writer.writerow(["Net Mean", "Median", "Max", "Min", "STD", "Variance", "Range", "Skweness", "Kurtosis"])
		for batch_num in range(batchCnt*trialCnt):
			for col_num in range(total_cols):
				print("Row from %d to %d" % (batchSize*batch_num+1, (batch_num+1)*batchSize))
				print("Col %d" % (col_num))
				# Call the propExtraction method
				final_calculated_values = prop.propExtraction(file_list, batch_num, col_num, batchSize)
				writer.writerow(final_calculated_values)
				final_calculated_values[:] = []

#call the main method which will be executed at the start of program
main()
