# Import library
import cPickle 
import numpy as np
import csv
import sys

# Declare Global Constants
TRIAL = 40
ELEC = 40
SAMPLE = 8064
LABEL = 4
DEBUG = False

# Define Global Macros
dat = [TRIAL * SAMPLE *[ELEC*[0]]]
label =  [TRIAL *[LABEL*[0]]]

# Function definition 

#
# Debug print logic
#
def log(s):
    if DEBUG:
        print s

#
# Load datset from .dat file
#
def loadData(FileName):
	dataset = cPickle.load(open(FileName, 'rb'))
	return dataset

#
# Save "Data" & "Label" array information in seperate csv file
#

def extractArray(srcFile, dataType, dataset):
	with open(srcFile, 'wb') as datafile:
		FData = csv.writer(datafile, quoting=csv.QUOTE_ALL)
		for trial in range(0, TRIAL):			
			if dataType is "data":
				for sample in range(0, SAMPLE):
					for elec in range(0, ELEC):
						dat[0][trial*SAMPLE+sample][elec] = dataset['data'][trial][elec][sample]
						log("dat[%d][%d] = %d "%(trial*SAMPLE+sample, elec, dat[0][trial*SAMPLE+sample][elec]))
					log(dat[0][trial*SAMPLE+sample][2])
					FData.writerow(dat[0][trial*SAMPLE+sample])
			else:
				for label_ in range(0, LABEL):
					label[0][trial][label_] = dataset['labels'][trial][label_]
					log("label[%d][%d] = %d "%(trial, label_, label[0][trial][label_]))
				FData.writerow(label[0][trial])
			log("\n")
	datafile.close()

#
# Main() start from here
#
def main():
	log("Load data of %s" %(sys.argv[1]))
	dataset = loadData(sys.argv[1])
	log("extract data array from %s" %(sys.argv[1]))
	extractArray("data_%02d.csv" %(int(filter(str.isdigit, sys.argv[1]))), "data", dataset)
	log("extract label array from %s" %(sys.argv[1]))
	extractArray("label_%02d.csv" %(int(filter(str.isdigit, sys.argv[1]))), "label", dataset)

main()
