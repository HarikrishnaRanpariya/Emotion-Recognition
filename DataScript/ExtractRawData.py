from __future__ import division
import os 
import sys
from os import walk

def main():
	inputfilePath = sys.argv[1]
	f = []
	for (dirpath, dirnames, filenames) in walk(inputfilePath):
		filenames = [i for i in filenames if i.endswith('.dat')] 
		f.extend(filenames)
		break
	
	i=0
	for fileName in f:
		
		inputFile = inputfilePath+fileName
		
		datafile = (".\data\data_%s.csv" %(fileName.split('.')[0][1:3]))
		labelfile = (".\label\label_%s.csv" %(fileName.split('.')[0][1:3]))
		
		
		os.system('python 1_datToCSV_Coverter.py %s' %inputFile)
		print("Converted raw data from %s to %s & %s" %(fileName, datafile, labelfile))
	
main()

