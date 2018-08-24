from __future__ import division
from os import walk
import os 
import sys
import errno

def main():
	inputfilePath = sys.argv[1]
	f = []
	for (dirpath, dirnames, filenames) in walk(inputfilePath):
		filenames = [i for i in filenames if i.endswith('.csv')] 
		f.extend(filenames)
		break
	
	i=0
	for datafile in f:
		
		propfile = (".\property\prop_%s.csv" %(datafile.split('.')[0][5:7]))
		if not os.path.exists(os.path.dirname(propfile)):
			try:
				os.makedirs(os.path.dirname(propfile))
			except OSError as exc: # Guard against race condition
				if exc.errno != errno.EEXIST:
					raise
		
		mergefile = (".\merge\merge_%s.csv" %(datafile.split('.')[0][5:7]))
		if not os.path.exists(os.path.dirname(mergefile)):
			try:
				os.makedirs(os.path.dirname(mergefile))
			except OSError as exc: # Guard against race condition
				if exc.errno != errno.EEXIST:
					raise
			
		modelfile = (".\modeldata\model_%s.csv" %(datafile.split('.')[0][5:7]))
		if not os.path.exists(os.path.dirname(modelfile)):
			try:
				os.makedirs(os.path.dirname(modelfile))
			except OSError as exc: # Guard against race condition
				if exc.errno != errno.EEXIST:
					raise
		
		print(propfile)
		print(mergefile)
		print(modelfile)
		
		os.system('python 2_Property.py %s %s' %(inputfilePath+datafile, propfile))
		print("Properties are calculated: from %s to %s " %(inputfilePath+datafile, propfile))
		
		os.system('python 3_PropMerger.py %s %s' %(propfile, mergefile))
		print("Property Merger is done: from %s to %s " %(propfile, mergefile))
		
		os.system('python 4_ModelDataCoverter.py %s %s' %(mergefile, modelfile))
		print("Coverted to Model data: from %s to %s " %(mergefile, modelfile))

main()

