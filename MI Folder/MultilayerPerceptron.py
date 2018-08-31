import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

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
        self.n_input    = n_input
        self.n_classes  = n_classes
    
    #Store layers weight & bias
    def ModelGraph(self, x):
    
        #create model    
        weights = {
        	'h1' : tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1])),
        	'h2' : tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
        	'h3' : tf.Variable(tf.random_normal([self.n_hidden_2, self.n_hidden_3])),	
        	'out' : tf.Variable(tf.random_normal([self.n_hidden_3, self.n_classes])),
        }
        
        biases = {
        	'b1' : tf.Variable(tf.random_normal([self.n_hidden_1])),
        	'b2' : tf.Variable(tf.random_normal([self.n_hidden_2])),
        	'b3' : tf.Variable(tf.random_normal([self.n_hidden_3])),
        	'out' : tf.Variable(tf.random_normal([self.n_classes])),
        }
    
        #hidden fully connected layer with 4000 neurons
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        
        #applying ReLu   & dropout
        layer_1 = tf.nn.relu(layer_1)    
        drop_out_layer_1 = tf.nn.dropout(layer_1, 0.25) 
        
        #hidden fully connected layer with 500 neurons
        layer_2 = tf.add(tf.matmul(drop_out_layer_1, weights['h2']), biases['b2'])
        
        layer_2 = tf.nn.relu(layer_2)
        drop_out_layer_2 = tf.nn.dropout(layer_2, 0.50)
        
        #hidden fully connected layer with 1000 neurons
        layer_3 = tf.add(tf.matmul(drop_out_layer_2, weights['h3']), biases['b3'])
        
        layer_3 = tf.nn.relu(layer_3)
        drop_out_layer_3 = tf.nn.dropout(layer_3, 0.50)
        
        
        #output layer with 2 classes
        out_layer = tf.add(tf.matmul(drop_out_layer_3, weights['out']), biases['out'])
        
        #out_layer = tf.nn.softmax(out_layer)
        drop__out_out_layer = tf.nn.dropout(out_layer, 0.50)
        
        return drop__out_out_layer
    
    
    
    #training_function
    def train_model(self, X_train, X_test, y_train, y_test):
        # tf Graph input
        x = tf.placeholder("float",[None, self.n_input])
        y = tf.placeholder("float")
        prediction = self.ModelGraph(x)
        
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y_train))
        optim = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        optimizer = optim.minimize(cost)
        
        
        training_epochs = 25
        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            sess.run(init)
            
            for epoch in range(training_epochs):
                epoch_loss = 0
                #for i in range(0,31,8):
                
                _, c = sess.run([optimizer, cost], feed_dict={x:X_train, y:y_train})
                epoch_loss += c
                print('Epoch',epoch,'completed out of',training_epochs,'loss: ',epoch_loss)
            
            correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
            #NOTE Add accuracy of Arousal
            
            accuracy = tf.reduce_mean(tf.cast(correct,'float'))
            print('Accuracy:', accuracy.eval({x:X_test, y: y_test}))
        
def main():
    df1 = pd.read_csv('output.csv')
    df1_labels = pd.read_csv('labels.csv')
    
    df = df1.values.tolist()
    df_labels = df1_labels.values.tolist()
    
    
    obj = MultilayerPerceptron(0.00001, 8,1,4000,500, 1000, 2880, 2)
    #splitting dataset
    X_train, X_test, y_train, y_test = train_test_split(df, df_labels, test_size = 0.2)
    #parameters
    obj.train_model(X_train, X_test, y_train, y_test)

#
main()