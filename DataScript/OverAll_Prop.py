from __future__ import division
import csv
import pandas as pd
import statistics
import fileinput
import math
import sys

class PropMerger:
	"""This class derives on additional set of properties.
	
	With this class we are calculating an additional set of properties.
	Where the input data will be the data of all 7 batches which we used
	in 'Property class' to calculate those 9 values (net mean,median etc.)
	That data when combined produces it's own set of 9 overall values.
	
	Args:
	batchCnt : total number of batches (7 here)
	sensorCnt : total number of sensors (40 here)
	sampleCnt : total number of trials or videos (40 here)
	propCnt : total number of properties (9 here)
	
	
	"""
	
	def __init__(self, batchCnt, sensorCnt, sampleCnt, propCnt):
		"""Inits PropMerger class attributes
		
		Args:
		
		batchCnt : It is the total number of batches into which we are dividing 
		each electrode's reading values.
		sensorCnt : It's the total number of sensors.
		sampleCnt : It's the total number of trials or videos.
		propCnt : total number of properties.
	
		"""
		
		self.batchCnt=batchCnt
		self.sensorCnt=sensorCnt
		self.sampleCnt=sampleCnt
		self.propCnt=propCnt
		self.batchSize=0
		self.file_list = []
		
		# create a 3-D array to iterate over data and initialize it
		self.FProp = [[[0 for x in range(self.sensorCnt)] for y in range(self.propCnt)] for y in range(self.sampleCnt)]
		for trial in range(self.sampleCnt):
			for prop in range (self.propCnt):
				for sensor in range(self.sensorCnt):
					self.FProp[trial][prop][sensor] = []
	
	def loadData(self, inputFile):
		"""Loads the data from file
		
		This method loads the data from inputFile path and appends it 
		to file_list
		
		Args:
		inputFile: It is the data file.
		
		
		"""
		
		# printing file data
		print("\nFile data is:")
		# Append the data to file_list list
		filereader = csv.reader(open(inputFile), delimiter =',')
		for row in filereader:
			self.file_list.append(row)
		
		# Get the number of rows and columns to calculate the batchSize
		total_rows = len(self.file_list)
		total_cols = len(self.file_list[0])
		self.batchSize =  total_rows/(self.batchCnt*self.sampleCnt)
		
		print(self.file_list)
		print(self.file_list[0])
		print(total_rows)
		print(total_cols)
		
		# Store the data in form of a list in a three dimensional list
		for trial in range (self.sampleCnt):
			for prop in range (self.propCnt):
				for sensor in range(self.sensorCnt):
					for batch in range(self.batchCnt):
						self.FProp[trial][prop][sensor].append(self.file_list[(trial*(self.batchCnt*self.sensorCnt))+(batch*self.sensorCnt+sensor+1)][prop])
	
	
	def findMean(self):
		"""Calculates overall mean
		
		This method calculates overall mean of all 7 batches data
		
		Args:
		There are no separate arguments.It just accesses FProp list and
		calculates & appends the final mean value.
		
		Returns:
		It returns the FProp list with overall batch data mean values 
		stored in it.
		
		"""
		
		for trial in range(self.sampleCnt):
			for sensor in range(self.sensorCnt):
				temp = 0.0
				for batch in range(self.batchCnt):
					temp += float(self.FProp[trial][0][sensor][batch])
				self.FProp[trial][0][sensor].append(temp/self.batchCnt)
	
	def findMedian(self):
		"""Calculates overall Median
		
		This method calculates overall median of all 7 batches data
		
		Args:
		There are no separate arguments.It just accesses FProp list and
		calculates & appends the final median value.
		
		Returns:
		It returns the FProp list with overall batch data median values 
		stored in it.
		
		"""
		
		median = []
		for trial in range(self.sampleCnt):
			for sensor in range(self.sensorCnt):
				for batch in range(self.batchCnt):
					median.append(self.FProp[trial][1][sensor][batch])
				# Sort the obtained median data
				median.sort()
				# find the mid-point
				mid_point = int(self.batchCnt/2)
				if(mid_point%2==0):       
					final_median = (float(median[mid_point-1])+float(median[mid_point]))/2
					self.FProp[trial][1][sensor].append(float(final_median))
				else:
					self.FProp[trial][1][sensor].append(float(median[mid_point]))
				median = []
	
	def fingSTD(self):
		"""Calculates overall Standard Deviation
		
		This method calculates overall Standard Deviation of all 7 batches data
		
		Args:
		There are no separate arguments.It just accesses FProp list and
		calculates & appends the final Standard Deviation value.
		
		Returns:
		It returns the FProp list with overall batch data Standard 
		Deviation values stored in it.
		
		"""
		
		
		for trial in range(self.sampleCnt):
			for sensor in range(self.sensorCnt):
				Stemp=0.0
				for batch in range(self.batchCnt):
					Stemp+=math.pow(float(self.FProp[trial][4][sensor][batch]), 2)*(self.batchSize)
				if (self.batchSize*self.batchCnt) > 0:
					self.FProp[trial][4][sensor].append(math.sqrt(Stemp/(self.batchSize*self.batchCnt)))
	
	def findVariance(self):
		"""Calculates overall Variance
		
		This method calculates overall Variance of all 7 batches data
		
		Args:
		There are no separate arguments.It just accesses FProp list and
		calculates & appends the final Variance value.
		
		Returns:
		It returns the FProp list with overall batch data Variance values
		stored in it.
		
		"""
		# Variance = math.pow(std_dev, 2)
		for trial in range(self.sampleCnt):
			for sensor in range(self.sensorCnt):
				self.FProp[trial][5][sensor].append(math.pow(self.FProp[trial][4][sensor][7], 2))
	
	def findRange(self):
		"""Calculates overall Range
		
		This method calculates overall Range of all 7 batches data
		
		Args:
		There are no separate arguments.It just accesses FProp list and
		calculates & appends the final Range value.
		
		Returns:
		It returns the FProp list with overall batch data Range values
		stored in it.
		"""
		
		for trial in range(self.sampleCnt):
			for sensor in range(self.sensorCnt):
				self.FProp[trial][6][sensor].append(float(self.FProp[trial][2][sensor][7])-float(self.FProp[trial][3][sensor][7]))
	
	def findSkewness(self):
		"""Calculates overall Skewness
		
		This method calculates overall Skewness of all 7 batches data
		
		Args:
		There are no separate arguments.It just accesses FProp list and
		calculates & appends the final Skewness value.
		
		Returns:
		It returns the FProp list with overall batch data Skewness values
		stored in it.
		"""
		
		# Skewness = (3*(net_mean - final_median))/std_dev
		for trial in range(self.sampleCnt):
			for sensor in range(self.sensorCnt):
				self.FProp[trial][7][sensor].append((3*(float(self.FProp[trial][0][sensor][7]) - float(self.FProp[trial][1][sensor][7])))/float(self.FProp[trial][4][sensor][7]))
	
	def findKurtosis(self):
		"""Calculates overall Kurtosis value
		
		This method calculates overall Kurtosis value of all 7 batches data
		
		Args:
		There are no separate arguments.It just accesses FProp list and
		calculates & appends the final Kurtosis value.
		
		Returns:
		It returns the FProp list with overall batch data Kurtosis values
		stored in it.
		"""
		# Kurtosis = kurtosis_sum / math.pow(std_dev, 4)
		for trial in range(self.sampleCnt):
			for sensor in range(self.sensorCnt):
				Ktemp=0.0
				for batch in range(self.batchCnt):
					Ktemp += float(self.FProp[trial][8][sensor][batch])*self.batchSize*math.pow(float(self.FProp[trial][4][sensor][batch]),4)
				self.FProp[trial][8][sensor].append(Ktemp/(math.pow(float(self.FProp[trial][4][sensor][7]),4)*self.batchCnt*self.batchSize))
	
	def saveData(self, outputFile):
		""".csv file is generated
		
		From the FProp list we create a .csv file to save the data in file
		
		Args:
		outputFile: File on which data is going to be written
		
		Returns:
		A .csv file with FProp data
		
		"""
		
		# Copy final data (self.FProp[Prop][sensor][batch]) into a .csv file
		with open(outputFile,"w") as my_final_file:
			writer = csv.writer(my_final_file, lineterminator='\n')
			writer.writerow(["Batch1","Batch2","Batch3","Batch4","Batch5","Batch6","Batch7","OverAll"])
			for trial in range(self.sampleCnt):
				for sensor in range(self.sensorCnt):
					for prop in range(self.propCnt):
						writer.writerow(self.FProp[trial][prop][sensor])
	
	def calAvgProp(self, inputFile, outputFile):
		"""Call above defined functions
		
		This method is used to call above defined functions and to 
		calculate all properties and then save them to output file
		
		Args:
		inputFile: This is the file from which we are going to take or
		load the data
		outputFile : Final .csv file on which we are the data
		
		"""
		
		# Call respective methods
		self.loadData(inputFile)
		self.findMean()
		self.findMedian()
		# Calculate and append the max value to FProp
		for trial in range(self.sampleCnt):
			for sensor in range(self.sensorCnt):
				self.FProp[trial][2][sensor].append(max(list(map(float,self.FProp[trial][2][sensor]))))

		# Calculate and append the min value to FProp
		for trial in range(self.sampleCnt):
			for sensor in range(self.sensorCnt):
				self.FProp[trial][3][sensor].append(min(list(map(float,self.FProp[trial][3][sensor]))))
		
		#Call the respective functions to calculate the properties
		self.fingSTD()
		self.findVariance()
		self.findRange()
		self.findSkewness()
		self.findKurtosis()
		
		# Save the data into the output file
		self.saveData(outputFile)

def main():
	# main function
	
	# Get the inputfile argument from command line
	inputFilePath = sys.argv[1] # .\prop\prop_xx.csv
	# Get the output file name from command line
	outputFilePath = sys.argv[2] # .\merge\merge_xx.csv
	
	# pass the parameter values for final calculation
	batchCnt=7
	sensorCnt=40
	sampleCnt=40
	propCnt=9
	
	# create the PropMerger object and pass the parameters
	prop = PropMerger(batchCnt, sensorCnt, sampleCnt, propCnt)
	# call the calAvgProp method with input and output file paths
	prop.calAvgProp(inputFilePath, outputFilePath)

# call the main function. It will be executed before every thing else at the
# execution time
main()
