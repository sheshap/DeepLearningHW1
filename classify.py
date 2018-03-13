# This code helps to classify images in CIFAR-10 dataset using neural network classifier
# Derived inspiration from the Code: https://github.com/TapanBhavsar/image-classification-CIFAR-10-using-tensorflow
# The updated code is used as part of a homewrok assignment.
# Course : CIS 700 Advances in Deep Learning
import sys
import os
import numpy as np
import cv2
import argparse
from keras.datasets import cifar10
from keras.utils import np_utils
import tensorflow as tf

n_hidden_1 = 1024 # Features in layer 1
n_hidden_2 = 600 # Features in layer 2
n_input = 3072 # CIFAR-10 data input (img shape: 32*32)
n_classes = 10 # CIFAR-10 total classes 
# classes of images in CIFAR-10 dataset
class_name = ["aeroplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

# image class prediction function
def classify_name(predicts):
    max = predicts[0,0]
    temp = 0
    for i in range(len(predicts[0])):
        #highest probable class is chosen
        if predicts[0,i]>max:
                max = predicts[0,i]
                temp = i
    # print the class name
    print(class_name[temp])

def iterate_minibatches(inputs, targets, batchsize, shuffle=True):
    assert len(inputs) == len(targets)
    # shuffle is used in train the data
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def multilayer_perceptron(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# main function where network train and predict the output on random image
def main_function(num_epochs=10):
    # initialize input data shape and datatype for data and labels
    x = tf.placeholder(tf.float32,[None,3072])
    y = tf.placeholder(tf.int32,[None,10])
    #weights and bias used are randomly set
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    # call neural network classifier model
    predict = multilayer_perceptron(x,weights,biases)
    out_predict = tf.nn.softmax(predict)
    # error is propogated
    error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = predict,labels = y))
    optm = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(error) # adam optimizer is used for optimization
    corr = tf.equal(tf.argmax(predict,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(corr,tf.float32))
    # initialize saver for saving weight and bias values
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    if not os.path.exists("model/model.ckpt") and sys.argv[1]=="train":
        # initialize tensorflow session
        sess = tf.Session()
        sess.run(init)
        # load dataset using keras and dividing the dataset into train and test
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        X_train = X_train.reshape(-1, 3072).astype('float32')
        X_test = X_test.reshape(-1, 3072).astype('float32')
        X_train = X_train / 255
        X_test = X_test / 255
        y_train = np_utils.to_categorical(y_train, num_classes=10)
        y_test = np_utils.to_categorical(y_test, num_classes=10)
        # training will start
        bsize=100 #batch size used
        print("Loop\t Train Loss\t Train Acc % \t Test Loss \t \tTest Acc %\n")
        for epoch in range(num_epochs):
            train_err = 0
            train_acc = 0
            train_batches = 0
            # devide data into mini batch
            for batch in iterate_minibatches(X_train, y_train, bsize, shuffle=True):
                inputs, targets = batch
                # this is update weights
                sess.run([optm],feed_dict = {x: inputs,y: targets})
                # cost function
                err,acc= sess.run([error,accuracy],feed_dict = {x: inputs,y: targets})
                train_err += err
                train_acc += acc
                train_batches += 1
                
            test_err = 0
            test_acc = 0
            test_batches = 0
            # divide validation data into mini batch without shuffle
            for batch in iterate_minibatches(X_test, y_test, bsize, shuffle=False):
                inputs, targets = batch
                sess.run([optm],feed_dict = {x: inputs,y: targets}) # this is update weights
                err, acc = sess.run([error,accuracy],feed_dict = {x: inputs,y: targets}) # cost function
                test_err += err
                test_acc += acc
                test_batches += 1
            # Current epoch with total number of epochs, training and testing loss with training and testing accuracy
            print("{}/{}\t\t{:.3f}\t\t{:.2f}\t\t{:.3f}\t\t{:.2f}\n".format(epoch + 1, num_epochs,train_err / (train_batches*bsize), train_acc/train_batches * 100,test_err / (test_batches*bsize),(test_acc / test_batches * 100)))
            
        # save weights values in ckpt file in given folder path
            save_path = saver.save(sess,"model/model.ckpt")

    #below portion of the code uses already trained model to test/predict
    elif (sys.argv[1]=="test" or sys.argv[1]=="predict"):
        sess = tf.Session()
        sess.run(init)
        #restore weights value for this neural network
        saver.restore(sess,"model/model.ckpt")
        # test the trained model using a random image (recommended to use 32x32 image
        img_s = sys.argv[2]#contains file name of the image used for testing
        img = cv2.imread(img_s)# read the image using opencv
        new_img = cv2.resize(img,dsize = (32,32),interpolation = cv2.INTER_CUBIC)
        new_img = np.asarray(new_img, dtype='float32') / 255
        img_ = new_img.reshape((-1, 3072))
        # output prediction for above image it gives 10 numeric numbers with it's class probability
        prediction = sess.run(out_predict,feed_dict={x: img_})
        # print predicted sclass
        classify_name(prediction)
        sess.close()
    else:
        print("Enter correct arguments")
# main function call
if __name__ == '__main__':
    main_function()
