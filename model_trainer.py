from training_data import TrainingData

class ModelTrainer(object):

    @classmethod
    def train(cls):
        td = TrainingData()
        td.pre_process()
        logits = cls.LeNet(td.x_train,6,16)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
        loss_operation = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate = rate)
        training_operation = optimizer.minimize(loss_operation)