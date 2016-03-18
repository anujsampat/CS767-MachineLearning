#
#   CS767 Machine Learning
#
#   Anuj Sampat
#   asampat@bu.edu
#
#   Python code to create a Long Short Term Memory (LSTM) Neural Network model
#   using the TensorFlow machine learning library.
#
#  Partly based on https://github.com/sherjilozair/char-rnn-tensorflow/blob/master/model.py
#
#


import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq
import numpy as np

# LSTM Model
class LSTMModel():
    def __init__(self, args, predict=False):

        self.args = args 
        if predict:
            batchSize = 1
            numSteps = 1

        # Various parameters for the LSTM. 
        # Hardcoded here for now.
        numSteps = 50 # Steps to unroll for
        batchSize = 50
        rnnSize = 128
        numLayers = 2
        gradClip = 5
        learningRate = 0.002
        decayRate = 0.97

        #Create LSTM layer and stack multiple layers. 
        lstmCell = rnn_cell.BasicLSTMCell(rnnSize)
        lstmNet = rnn_cell.MultiRNNCell([lstmCell] * numLayers)

        #Define placeholders.
        self.inputData = tf.placeholder(tf.int32, [batchSize, numSteps])
        self.targetOutput = tf.placeholder(tf.int32, [batchSize, numSteps])
        self.initialState = lstmNet.zero_state(batchSize, tf.float32)

        # If rnn_decoder is told to loop, this function will return to it the output at time
        # 't' for feeding as the input at time 't+1'. During training, this is generally
        # not done because we want to feed the *correct* input at all times and not what
        # is output. During prediction/testing, we loop the output back to the input to
        # generate our sequence of notes. 
        def feedBack(prev, _):
            prev = tf.nn.xw_plus_b(prev, softmax_w, softmax_b)
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        with tf.variable_scope('nn_lstm'):
            softmax_w = tf.get_variable("softmax_w", [rnnSize, args.vocabSize])
            softmax_b = tf.get_variable("softmax_b", [args.vocabSize])
            with tf.device("/cpu:0"):
                embedding = tf.get_variable("embedding", [args.vocabSize, rnnSize])
                inputs = tf.split(1, numSteps, tf.nn.embedding_lookup(embedding, self.inputData))
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
   
        #Call seq2seq rnn decoder.
        outputs, states = seq2seq.rnn_decoder(inputs, self.initialState, lstmNet, loop_function=feedBack if predict else None, scope='nn_lstm')
        output = tf.reshape(tf.concat(1, outputs), [-1, rnnSize])

        #Logit and probability
        #softmax_w = tf.get_variable("softmax_w", rnnSize, [args.vocabSize])
        #softmax_b = tf.get_variable("softmax_b", [args.vocabSize])
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)

        # Calculate loss compared to targetOutput
        loss = seq2seq.sequence_loss_by_example([self.logits],
                [tf.reshape(self.targetOutput, [-1])],
                [tf.ones([batchSize * numSteps])],
                args.vocabSize)

        # Set the cost to minimize total loss.
        self.cost = tf.reduce_sum(loss)

        # Learning rate remains constant (not trainable)
        self.finalState = states[-1]
        self.learningRate = tf.Variable(0.0, trainable=False)

        # Define gradient and trainable variables for adjusting 
        # during training/optimization.
        trainableVars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainableVars),
                gradClip)

        # We use the Adam optimizer.
        #optimizer = tf.train.GradientDescentOptimizer(self.learningRate).minimize(loss)
        #optimizer = tf.train.AdagradOptimizer(self.learningRate, initial_accumulator_value=0.1)
        #self.trainStep = optimizer.apply_gradients(zip(grads, trainableVars))
        optimizer = tf.train.AdamOptimizer(self.learningRate)
        self.trainStep = optimizer.apply_gradients(zip(grads, trainableVars))


    #
    #  From https://github.com/sherjilozair/char-rnn-tensorflow/blob/master/model.py
    #
    def sample(self, sess, chars, vocab, num=200, prime='The '):
        state = self.cell.zero_state(1, tf.float32).eval()
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state:state}
            [state] = sess.run([self.finalState], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        char = prime[-1]
        for n in xrange(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state:state}
            [probs, state] = sess.run([self.probs, self.finalState], feed)
            p = probs[0]
            # sample = int(np.random.choice(len(p), p=p))
            sample = weighted_pick(p)
            pred = chars[sample]
            ret += pred
            char = pred
        return ret
