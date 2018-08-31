import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
import sys
from os import walk

class MultilayerPerceptron:
	
	"""This class implements Deep Neural Network.
	
	MultilayerPerceptron classes consists of the implementation of 
	Multi-layer Deep Neural Network for the classification of DEAP EEG 
	dataset.
	The class have three methods in total(including the __init__ method)
	
	Class Attributes:
	learning_rate : learning_rate holds the value of the rate with which
	the gradient is gonna change at the time of weights & biases optimi-
	-zation using AdamOptimizer.
	
	batch_size : batch_size holds the number of files or batches we are 
	gonna use for the training part of the model,here we are going with 
	32 batches.
	
	n_hidden_1 : the number of nodes in the first hidden layer.
	n_hidden_2 : the total number of nodes in the second hidden layer.
	n_hidden_3 : the number of nodes in the third hidden layer.
	n_input : total number of input layer nodes.
	n_classes : number of nodes in the final layer of the model.
	n_epochs : the number of iteration or epochs for which model will be
	executed in every batch.
	
	"""
	
	def __init__(self, learning_rate, batch_size, n_hidden_1, n_hidden_2, n_hidden_3, n_input, n_classes, n_epochs):
		#(0.00001, 8,1,4000,500, 1000, 2880, 2)
		
		"""Inits MultilayerPerceptron class with learning_rate,batch_size,
		n_hidden_1,n_hidden_2,n_input,n_classes,n_epochs.
		"""
		self.learning_rate = learning_rate
		
		self.batch_size = batch_size
		#self.display_step = display_step
		
		#network parameters
		self.n_hidden_1 = n_hidden_1
		self.n_hidden_2 = n_hidden_2
		self.n_hidden_3 = n_hidden_3
		self.n_input	= n_input
		self.n_classes  = n_classes
		self.n_epochs 	= n_epochs
		self.weights = {
			'h1' : tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1])),
			'h2' : tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
			'h3' : tf.Variable(tf.random_normal([self.n_hidden_2, self.n_hidden_3])),
			'out' : tf.Variable(tf.random_normal([self.n_hidden_3, self.n_classes])),
		}
		
		self.biases = {
			'b1' : tf.Variable(tf.random_normal([self.n_hidden_1])),
			'b2' : tf.Variable(tf.random_normal([self.n_hidden_2])),
			'b3' : tf.Variable(tf.random_normal([self.n_hidden_3])),
			'out' : tf.Variable(tf.random_normal([self.n_classes])),
		}
	
	#Store layers weight & bias
	def ModelGraph(self):
		"""Implements the computation graph
		
		This method creates or defines the computation graph structure.
		Layer definitions as well as placeholder definitions are listed 
		in this method.
		
		Args:
		ModelGraph takes no arguments. All arguments are accessed from 
		the class attributes,which are intialized in the __init__ method
		
		Returns:
		It returns the final or the output layer after applying the drop-
		-out.
		
		"""
		
		#create model
		self.x = tf.placeholder("float", [None, self.n_input])
		self.y = tf.placeholder("float", [None, self.n_classes])
		
		#hidden fully connected layer with 4000 neurons
		layer_1 = tf.add(tf.matmul(self.x, self.weights['h1']), self.biases['b1'])
		
		#applying ReLu   & dropout
		layer_1 = tf.nn.relu(layer_1)
		drop_out_layer_1 = tf.nn.dropout(layer_1, 0.25) 
		
		#hidden fully connected layer with 500 neurons
		layer_2 = tf.add(tf.matmul(drop_out_layer_1, self.weights['h2']), self.biases['b2'])
		
		layer_2 = tf.nn.relu(layer_2)
		drop_out_layer_2 = tf.nn.dropout(layer_2, 0.50)
		
		#hidden fully connected layer with 1000 neurons
		layer_3 = tf.add(tf.matmul(drop_out_layer_2, self.weights['h3']), self.biases['b3'])
		
		layer_3 = tf.nn.relu(layer_3)
		drop_out_layer_3 = tf.nn.dropout(layer_3, 0.50)
		
		#output layer with 2 classes
		out_layer = tf.add(tf.matmul(drop_out_layer_3, self.weights['out']), self.biases['out'])
		
		#out_layer = tf.nn.softmax(out_layer)
		drop__out_out_layer = tf.nn.dropout(out_layer, 0.50)
		
		return drop__out_out_layer
	
	
	#training_function
	def train_model(self, prediction, DataFilePath, DataFile, LabelFilePath, LabelFile):#prediction, X_train, X_test, y_train, y_test):
		"""train_model holds the training as well as testing part.
		
		train_model takes the data and label files as the input arguments
		and is responsible for the training as well as testing part of the
		model. The final accuracy is also calculated in this part.
		
		Args:
		prediction : this argument points to the modelgraph method
		defined above and holds the final output_layer values.
		DataFilePath : this argument holds the data file address in the 
		directory.
		Datafile : this is a list type argument & it holds the names of 
		the data files.
		LabelFilePath : this argument holds the address of the labels file
		in the directory.
		LabelFile : This is list type argument and it holds the names of
		all label files.
		
		Returns:
		It doesn't return anything at the end. All work is done during 
		it's execution.
		
		"""
		
		# tf Graph input
		Flag=True
		total_Accuracy = 0.00
		
		#prediction = self.ModelGraph(x)
		crossEntropy = tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = self.y)
		'''
		softmax = tf.nn.softmax(prediction)
		m = self.y.shape[0]
		log_likelihood = -np.log(softmax[range(m),self.y])
		crossEntropy = np.sum(log_likelihood) / m
		'''
		cost = tf.reduce_mean(crossEntropy)
		optim = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
		optimizer = optim.minimize(cost)
		
		
		self.n_epochs = 50
		init = tf.global_variables_initializer()
		
		# configures the tensorflow to use CPU in place of GPU during 
		# session run.
		config = tf.ConfigProto(device_count = {'GPU':0})
		
		# session variable is created for using computation graph
		with tf.Session(config = config) as sess:
			sess.run(init)
			# loop for traversing the datafiles
			for Tindex in range(len(DataFile)):
				Flag=True
				
				for index in range(len(DataFile)):
					# Create a two dimensional list for storing valence
					# & arousal values from the dataset.
					labelsList= [[0 for x in range(2)] for y in range(40)]
					
					# get the data and labels files
					data = pd.read_csv(DataFilePath+DataFile[index])
					labels = pd.read_csv(LabelFilePath+LabelFile[index])
					
					# convert data and labels files into lists
					RawdataList = data.values.tolist()
					RawlabelsList = labels.values.tolist()
					# Read Label data from the file.& convert them to 
					# one-hot encoding
					for i in range(len(RawlabelsList)):
						for j in range(2):
							if RawlabelsList[i][j] > 4:
								labelsList[i][j] = 1
								
							else:
								labelsList[i][j] = 0
								
					# create the test file
					if Tindex == index:
						X_test=RawdataList
						y_test=labelsList
						print("--%s" %DataFile[Tindex])
					# else create the datafile and append
					else:
						if Flag == True:
							X_train = RawdataList
							y_train = labelsList
							Flag=False
						else:
							X_train += RawdataList
							y_train += labelsList
							#print("++")
					del labelsList
				
				# start the epochs loop
				for epoch in range(self.n_epochs):
					# get the optimizer and cost values from the comput-
					# -ation grapha and store them in _ and c respectively.
					_, c = sess.run([optimizer, cost], feed_dict={self.x:X_train, self.y:y_train})
					
					# print the epoch and loss values
					print('Epoch',epoch,'completed out of',self.n_epochs,'loss: ',c)
				
				# apply the softmax on the final result of the graph 
				pred = tf.nn.softmax(prediction)
				#compare the softmax output with groundtruths 'y'
				correct = tf.equal(tf.argmax(pred,1), tf.argmax(self.y,1))
				
				# calculate the accuracy value
				accuracy = tf.reduce_mean(tf.cast(correct,'float'))
				print('Accuracy:', accuracy.eval({self.x:X_test, self.y: y_test}))
				
				# get the total accuracy in total_accuracy variable
				total_Accuracy += accuracy.eval({self.x:X_test, self.y: y_test})
		print("Final accuracy is %d" %float(total_Accuracy/(len(DataFile))))
		
		
def main():
	"""Main function is executed first
	
	Main function is called and executed at the start of the program and
	it then further passes the arguments to the multilayer_perceptron 
	class for execution.
	
	Args:
	Takes the data and labels folder names at the time of execution from
	command-line
	
	Returns:
	Nothing is returned only calls are made to class.
	
	"""
	
	# get the data files folder name from argv list which holds the 
	# command-line arguments
	DataFilePath = sys.argv[1]
	# get the labels file folder name from command-line list.
	LabelFilePath = sys.argv[2]
	# create a blank list DataFile which will store file-names
	DataFile = []
	
	# get the directory path,names and filenames from the datafilepath
	# that is the folder name we passed in the argument line
	for (dirpath, dirnames, filenames) in walk(DataFilePath):
		# only pick .csv files from all folder files
		filenames = [i for i in filenames if i.endswith('.csv')] 
		#append them to Datafile names list
		DataFile.extend(filenames)
		break
	
	# similarly create a list to store label file names
	LabelFile = []
	
	#walk throught the Labels folder to collect the filenames
	for (dirpath, dirnames, filenames) in walk(LabelFilePath):
		# pick .csv files
		filenames = [i for i in filenames if i.endswith('.csv')] 
		# append them to list
		LabelFile.extend(filenames)
		break
	
	# create the object of MultilayerPerceptron class and pass the 
	# required arguments.
	obj = MultilayerPerceptron(0.00001, 8, 5500,1500, 2500, 2880, 2,100)
	
	# call the ModelGraph method to obtain the prediction pointer or 
	# values.
	prediction = obj.ModelGraph()
	
	# call the train_model method with the required arguments
	#obj.train_model(prediction, X_train, X_test, Y_train, Y_test)
	obj.train_model(prediction, DataFilePath, DataFile, LabelFilePath, LabelFile)

# call the main() function to start the execution.Main method will be the
# first to execute when we start our program.
main()
