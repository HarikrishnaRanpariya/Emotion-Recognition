from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from os import walk
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
import sklearn.datasets
import sklearn
import numpy as np
import pandas as pd
import sys
import os

class MultilayerPerceptron:
	
	def __init__(self, learning_rate, batch_size, display_step, n_hidden_1, n_hidden_2, n_hidden_3, n_input, n_classes):
		#(0.00001, 8,1,4000,500, 1000, 2880, 2)
		self.learning_rate = learning_rate
		
		self.batch_size = batch_size
		self.display_step = display_step
		
		#network parameters
		self.n_hidden_1 = n_hidden_1
		self.n_hidden_2 = n_hidden_2
		self.n_hidden_3 = n_hidden_3
		self.n_input	= n_input
		self.n_classes  = n_classes
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
	def train_model(self, prediction, X_train, X_test, y_train, y_test):
		
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = self.y))
		optim = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
		optimizer = optim.minimize(cost)
		
		training_epochs = 25
		init = tf.global_variables_initializer()
		# issue : 4 Added a workaround in form of CPU use
		config = tf.ConfigProto(device_count = {'GPU':0})
		
		with tf.Session(config = config) as sess:
			sess.run(init)
			
			for epoch in range(training_epochs):
				_, c = sess.run([optimizer, cost], feed_dict={self.x:X_train, self.y:y_train})
				print('Epoch',epoch,'completed out of',training_epochs,'loss: ',c)
			
			correct = tf.equal(tf.argmax(prediction,1), tf.argmax(self.y,1))
			
			accuracy = tf.reduce_mean(tf.cast(correct,'float'))
			print('Accuracy:', accuracy.eval({self.x:X_test, self.y: y_test}))
			
	
def main():
	Flag=True
	DataFilePath = sys.argv[1]
	LabelFilePath = sys.argv[2]
	DataFile = []
	for (dirpath, dirnames, filenames) in walk(DataFilePath):
		filenames = [i for i in filenames if i.endswith('.csv')] 
		DataFile.extend(filenames)
		break
	
	LabelFile = []
	for (dirpath, dirnames, filenames) in walk(LabelFilePath):
		filenames = [i for i in filenames if i.endswith('.csv')] 
		LabelFile.extend(filenames)
		break
	
	obj = MultilayerPerceptron(0.00001, 8,1,4000,500, 1000, 2880, 2)
	prediction = obj.ModelGraph()
	for Tindex in range(len(DataFile)):
		Flag=True
		
		MI_Outputfile = (".\MIOutput\output_%s.csv" %(DataFile[Tindex].split('.')[0][5:7]))
		if not os.path.exists(os.path.dirname(MI_Outputfile)):
			try:
				os.makedirs(os.path.dirname(MI_Outputfile))
			except OSError as exc: # Guard against race condition
				if exc.errno != errno.EEXIST:
					raise
		
		for index in range(len(DataFile)):
			labelsList= [[0 for x in range(2)] for y in range(40)]
			
			data = pd.read_csv(DataFilePath+DataFile[index])
			labels = pd.read_csv(LabelFilePath+LabelFile[index])
			
			dataList = data.values.tolist()
			RawlabelsList = labels.values.tolist()
			for i in range(len(RawlabelsList)):
				for j in range(2):
					if RawlabelsList[i][j] > 4:
						labelsList[i][j] = 1
					else:
						labelsList[i][j] = 0
						
			
			if Tindex == index:
				X_test=dataList
				Y_test=labelsList
				print("--%s" %DataFile[Tindex])
			else:
				if Flag == True:
					X_train = dataList
					Y_train = labelsList
					Flag=False
				else:
					X_train += dataList
					Y_train += labelsList
					#print("++")
			del labelsList
					
					
		obj.train_model(prediction, X_train, X_test, Y_train, Y_test)

main()
