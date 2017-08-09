import tensorflow as tf
from training_data import TrainingData
import numpy as np
from sklearn.utils import shuffle
import sys

class Model(object):
    def __init__(self):
        self.training_data=TrainingData.get_data()   # type : TrainingData

        self.network = None
        self.training_operation = None
        self.accuracy_operation = None
        self.session = tf.Session()
        self.saved_model = None


    def init(self):

        self.init_hyper_params()
        self.init_network_params()
        self.create_network()
        self.create_training_operation()
        self.create_accuracy_operation()

    def init_hyper_params(self):

        self.learning_rate=0.0005

    def init_network_params(self):
        self.p_features = tf.placeholder(tf.float32, [None, 32,32,1])
        self.p_labels = tf.placeholder(tf.float32, [None,43])

    def create_training_operation(self):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.network, labels=self.p_labels)
        loss_operation = tf.reduce_mean(cross_entropy)
        self.training_operation  = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(loss_operation)


    def create_accuracy_operation(self):
        correct_prediction = tf.equal(tf.argmax(self.network, 1), tf.argmax(self.p_labels, 1))
        self.accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



    def train(self):

        EPOCHS = 100
        BATCH_SIZE = 200
        N = self.training_data.x_train.shape[0]

        self.session.run(tf.global_variables_initializer())

        for epoch in range(EPOCHS):
            for sindex in range(0,N,BATCH_SIZE):
                eindex = sindex + BATCH_SIZE
                features,labels=self.training_data.x_train[sindex:eindex],self.training_data.y_train[sindex:eindex]
                self.session.run(self.training_operation,feed_dict={self.p_features:features,self.p_labels:labels})
                if ((sindex//BATCH_SIZE) % 10 == 0 ): print(".",end="",flush=True)


            features,labels=self.training_data.x_valid,self.training_data.y_valid
            a_output = self.session.run(self.accuracy_operation,feed_dict={self.p_features:features,self.p_labels:labels})
            print(":" , a_output)



    def predict(self):
        pass

    def get_trained_model(self):
        pass




    def create_network(self):
        mean = 0
        standard_deviation = 0.1
        dropout = 0.5
        #-----------------------------------------------------------------------------
        # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 32x32x8.
        filter_size,inp_channels,out_channels = 5,1,8
        weights=tf.Variable(tf.truncated_normal((filter_size,filter_size,inp_channels,out_channels),mean,standard_deviation))
        biases=tf.Variable(tf.zeros(out_channels,1))
        network=tf.nn.conv2d(self.p_features,weights,[1,1,1,1],'SAME')
        network=tf.nn.bias_add(network,biases)
        network=tf.nn.relu(network)
        # TODO: Pooling. Input = 32x32x8. Output = 16x16x8.
        network=tf.nn.max_pool(network,[1,2,2,1],[1,2,2,1],'VALID')

        #-----------------------------------------------------------------------------
        # TODO: Layer 2: Convolutional. Input = 16x16x8 , Output = 16x16x16.
        filter_size,inp_channels,out_channels = 5,8,16
        weights=tf.Variable(tf.truncated_normal((filter_size,filter_size,inp_channels,out_channels),mean,standard_deviation))
        biases=tf.Variable(tf.zeros(out_channels,1))
        network=tf.nn.conv2d(network,weights,[1,1,1,1],'SAME')
        network=tf.nn.bias_add(network,biases)
        network=tf.nn.relu(network)
        # TODO: Pooling. Input = 16x16x16. Output = 8x8x16.
        network=tf.nn.max_pool(network,[1,2,2,1],[1,2,2,1],'VALID')

        #-----------------------------------------------------------------------------
        # TODO: Flatten. Input = 4x4x32. Output = 512
        tensor_size = 8*8*16
        network = tf.contrib.layers.flatten(network,[1,tensor_size])

        #-----------------------------------------------------------------------------
        # TODO: Layer 4: Fully Connected. Input = 1024 . Output = 100.
        input_size, output_size =1024,512
        weights=tf.Variable(tf.truncated_normal((input_size,output_size),mean,standard_deviation))
        biases=tf.Variable(tf.zeros(output_size,1))
        network = tf.matmul(network,weights)
        network=tf.nn.bias_add(network,biases)
        network=tf.nn.relu(network)
        network=tf.nn.dropout(network,dropout)

        #-----------------------------------------------------------------------------
        # TODO: Layer 4: Fully Connected. Input = 512 . Output = 100.
        input_size, output_size = 512,256
        weights=tf.Variable(tf.truncated_normal((input_size,output_size),mean,standard_deviation))
        biases=tf.Variable(tf.zeros(output_size,1))
        network = tf.matmul(network,weights)
        network=tf.nn.bias_add(network,biases)
        network=tf.nn.relu(network)
        network=tf.nn.dropout(network,dropout)

        #-----------------------------------------------------------------------------
        # TODO: Layer 4: Fully Connected. Input = 256. Output = 96.
        input_size, output_size = 256,96
        weights=tf.Variable(tf.truncated_normal((input_size,output_size),mean,standard_deviation))
        biases=tf.Variable(tf.zeros(output_size,1))
        network = tf.matmul(network,weights)
        network=tf.nn.bias_add(network,biases)
        network=tf.nn.relu(network)
        network=tf.nn.dropout(network,dropout)

        #-----------------------------------------------------------------------------
        # TODO: Layer 5: Output Layer : Fully Connected. Input = 100. Output = 43.
        input_size, output_size = 96,43
        weights=tf.Variable(tf.truncated_normal((input_size,output_size),mean,standard_deviation))
        biases=tf.Variable(tf.zeros(output_size,1))
        network = tf.matmul(network,weights)
        network = tf.nn.bias_add(network,biases)

        #-----------------------------------------------------------------------------

        self.network=network


if __name__ == "__main__":

    model = Model()
    model.init()
    model.train()


