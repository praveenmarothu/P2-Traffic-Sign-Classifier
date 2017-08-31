import glob

import cv2
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import sys
import os
import pickle
from image_processor import ImageProcessor
from training_data import TrainingData

class Model(object):

    saved_model_session=None

    def __init__(self):
        self.training_data=TrainingData.get_data()   # type : TrainingData

        self.network = None
        self.training_operation = None
        self.accuracy_operation = None
        self.top_k_operation = None
        #init
        self.init_hyper_params()
        self.init_network_params()
        self.create_network()
        self.create_training_operation()
        self.create_accuracy_operation()
        self.create_top_k_operation()

    def init_hyper_params(self):

        self.learning_rate=0.001
        self.epochs=20
        self.batch_size=128

    def init_network_params(self):
        self.p_features = tf.placeholder(tf.float32, [None, 32,32,1])
        self.p_labels = tf.placeholder(tf.float32, [None,43])

    def create_training_operation(self):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.network, labels=self.p_labels)
        loss_operation = tf.reduce_mean(cross_entropy)
        self.training_operation  = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(loss_operation)

    def create_top_k_operation(self):
        softmax_logits = tf.nn.softmax(logits=self.network)
        self.top_k_operation = tf.nn.top_k(softmax_logits, k=3)

    def create_accuracy_operation(self):
        correct_prediction = tf.equal(tf.argmax(self.network, 1), tf.argmax(self.p_labels, 1))
        self.accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



    def create_network(self):
        mean = 0
        standard_deviation = 0.1
        dropout = 0.5
        #-----------------------------------------------------------------------------
        # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 32x32x8.
        filter_size,inp_channels,out_channels = 5,1,32
        weights=tf.Variable(tf.truncated_normal((filter_size,filter_size,inp_channels,out_channels),mean,standard_deviation))
        biases=tf.Variable(tf.zeros(out_channels,1))
        network=tf.nn.conv2d(self.p_features,weights,[1,1,1,1],'SAME')
        network=tf.nn.bias_add(network,biases)
        network=tf.nn.relu(network)
        # TODO: Pooling. Input = 32x32x8. Output = 16x16x8.
        network=tf.nn.max_pool(network,[1,2,2,1],[1,2,2,1],'VALID')

        #-----------------------------------------------------------------------------
        # TODO: Layer 2: Convolutional. Input = 16x16x8 , Output = 16x16x16.
        filter_size,inp_channels,out_channels = 5,32,64
        weights=tf.Variable(tf.truncated_normal((filter_size,filter_size,inp_channels,out_channels),mean,standard_deviation))
        biases=tf.Variable(tf.zeros(out_channels,1))
        network=tf.nn.conv2d(network,weights,[1,1,1,1],'SAME')
        network=tf.nn.bias_add(network,biases)
        network=tf.nn.relu(network)
        # TODO: Pooling. Input = 16x16x16. Output = 8x8x16.
        network=tf.nn.max_pool(network,[1,2,2,1],[1,2,2,1],'VALID')

        #-----------------------------------------------------------------------------
        # TODO: Layer 2: Convolutional. Input = 16x16x8 , Output = 16x16x16.
        filter_size,inp_channels,out_channels = 5,64,128
        weights=tf.Variable(tf.truncated_normal((filter_size,filter_size,inp_channels,out_channels),mean,standard_deviation))
        biases=tf.Variable(tf.zeros(out_channels,1))
        network=tf.nn.conv2d(network,weights,[1,1,1,1],'SAME')
        network=tf.nn.bias_add(network,biases)
        network=tf.nn.relu(network)
        # TODO: Pooling. Input = 16x16x16. Output = 8x8x16.
        network=tf.nn.max_pool(network,[1,2,2,1],[1,2,2,1],'VALID')

        #-----------------------------------------------------------------------------
        # TODO: Flatten. Input = 4x4x32. Output = 512
        tensor_size = 4*4*128
        network = tf.contrib.layers.flatten(network,[1,tensor_size])

        #-----------------------------------------------------------------------------
        # TODO: Layer 4: Fully Connected. Input = 1024 . Output = 100.
        input_size, output_size =2048,1024
        weights=tf.Variable(tf.truncated_normal((input_size,output_size),mean,standard_deviation))
        biases=tf.Variable(tf.zeros(output_size,1))
        network = tf.matmul(network,weights)
        network=tf.nn.bias_add(network,biases)
        network=tf.nn.relu(network)
        network=tf.nn.dropout(network,dropout)

        #-----------------------------------------------------------------------------
        # TODO: Layer 4: Fully Connected. Input = 512 . Output = 100.
        input_size, output_size = 1024,256
        weights=tf.Variable(tf.truncated_normal((input_size,output_size),mean,standard_deviation))
        biases=tf.Variable(tf.zeros(output_size,1))
        network = tf.matmul(network,weights)
        network=tf.nn.bias_add(network,biases)
        network=tf.nn.relu(network)
        network=tf.nn.dropout(network,dropout)


        #-----------------------------------------------------------------------------
        # TODO: Layer 5: Output Layer : Fully Connected. Input = 100. Output = 43.
        input_size, output_size = 256,43
        weights=tf.Variable(tf.truncated_normal((input_size,output_size),mean,standard_deviation))
        biases=tf.Variable(tf.zeros(output_size,1))
        network = tf.matmul(network,weights)
        network = tf.nn.bias_add(network,biases)

        #-----------------------------------------------------------------------------

        self.network=network



    def train(self):

        EPOCHS = self.epochs
        BATCH_SIZE = self.batch_size
        N = self.training_data.x_train.shape[0]

        session = tf.Session()
        session.run(tf.global_variables_initializer())

        for epoch in range(EPOCHS):
            for sindex in range(0,N,BATCH_SIZE):
                eindex = sindex + BATCH_SIZE
                features,labels=self.training_data.x_train[sindex:eindex],self.training_data.y_train[sindex:eindex]
                session.run(self.training_operation,feed_dict={self.p_features:features,self.p_labels:labels})
                if ((sindex//BATCH_SIZE) % 10 == 0 ): print(".",end="",flush=True)


            features,labels=self.training_data.x_valid,self.training_data.y_valid
            a_output = session.run(self.accuracy_operation,feed_dict={self.p_features:features,self.p_labels:labels})
            print(":" , a_output)

        saver = tf.train.Saver()
        saver.save(session, "saved_models/model1")

    def test(self):
        session = Model.get_saved_model_session()
        features,labels=self.training_data.x_test,self.training_data.y_test
        a_output = session.run(self.accuracy_operation,feed_dict={self.p_features:features,self.p_labels:labels})
        print("Testing Accuracy :" , a_output)

    def predict(self,imgs,labels):
        session = Model.get_saved_model_session()
        _imgs=[]

        for img in imgs: _imgs.append( ImageProcessor.grayscale_normalize(img) )

        imgs = np.array(_imgs)
        imgs = np.reshape(imgs,(-1,32,32,1) )

        values,indices = session.run(self.top_k_operation,feed_dict = {self.p_features:imgs})
        signnames = TrainingData.get_signnames()

        for idx,pset in enumerate(indices):
            print("")
            print( '=======================================================')
            print("Correct Sign :", labels[idx],"-",signnames[labels[idx]])
            print( '-------------------------------------------------------')
            print( '{0:7.2%} : {1: <2} - {2: <40}'.format(values[idx][0],pset[0],signnames[pset[0]]))
            print( '{0:7.2%} : {1: <2} - {2: <40}'.format(values[idx][1],pset[1],signnames[pset[1]]))
            print( '{0:7.2%} : {1: <2} - {2: <40}'.format(values[idx][2],pset[2],signnames[pset[2]]))

        print( '-------------------------------------------------------')

    @classmethod
    def get_saved_model_session(cls):

        if cls.saved_model_session != None:
            return cls.saved_model_session

        if not os.path.isfile("./saved_models/model1.meta"):
            model = Model()
            model.train()

        saver = tf.train.Saver()
        cls.saved_model_session = tf.Session()
        cls.saved_model_session.run(tf.global_variables_initializer())
        saver.restore(cls.saved_model_session, './saved_models/model1')
        return cls.saved_model_session




