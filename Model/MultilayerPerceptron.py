from __future__ import print_function

#importing data

import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

#getting data from files

file_name = 'our_csv_file.csv'

data = pd.read_csv(file_name)

#labels would be ratings column
y_labels = data.rating

#others columns would be used as features 
x_features = data.drop('rating', axis=1)

#splitting dataset
X_train, y_train, X_test, y_test = train_test_split(x_features, y_labels, test_size = 0.2)


#parameters
learning_rate = 0.00001
training_epochs = 250
batch_size = 101
display_step = 1

#network parameters
n_hidden_1 = 4000
n_hidden_2 = 500
n_hidden_3 = 1000
n_input    = 2960
n_classes  = 2

#tf Graph input
X = tf.placeholder("float", [None, n_input])
X = X_train
Y = tf.placeholder("float", [None, n_classes])
Y = y_train

#Store layers weight & bias

weights = {
	'h1' : tf.Variable(tf.random_normal([n_input, n_hidden_1])),
	'h2' : tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	'h3' : tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),	
	'out' : tf.Variable(tf.random_normal([n_hidden_3, n_classes])),
}

biases = {
	'b1' : tf.Variable(tf.random_normal([n_hidden_1])),
	'b2' : tf.Variable(tf.random_normal([n_hidden_2])),
	'b3' : tf.Variable(tf.random_normal([n_hidden_3])),
	'out' : tf.Variable(tf.random_normal([n_classes])),
}

#create model


def multilayer_perceptron(x):
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

    
#construct model
logits = multilayer_perceptron(X)

#define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss_op)

#initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    #Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = 7
        
        #loop over all batches
        for i in range(total_batch):
            batch_x = X_train/batch_size
            batch_y = y_train/batch_size
            
            #Run optimization op(backprop) and cost op (to get loss value)
            
            _, c = sess.run([train_op, loss_op], feed_dict={X:batch_x, Y:batch_y})
            
            #computing average loss
            avg_cost += c/total_batch
        
        #Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' %(epoch+1), "cost={:.9f}".format(avg_cost))
            
    print("Optimization Finished!")
    
    #TESTING TIME
    pred = tf.nn.softmax(logits)
    correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(Y,1))
    
    #CALCULATE ACCURACY
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
    print("Accuracy is :" %accuracy)
    
    
    