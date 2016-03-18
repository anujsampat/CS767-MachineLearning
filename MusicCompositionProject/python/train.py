#
#   CS767 Machine Learning
#
#   Anuj Sampat
#   asampat@bu.edu
#
#   Python code to train a Long Short Term Memory (LSTM) Neural Network
#   using the TensorFlow machine learning library.
#
#

import numpy as np
import tensorflow as tf
import os
import cPickle
from utils import TextLoader
from model import LSTMModel
import argparse
import time
import nltk
import collections


# Various parameters for the LSTM model
# Hardcoded here for now.
numSteps = 50 # Steps to unroll for
batchSize = 50
numEpochs = 50
learningRate = 0.002
decayRate = 0.97

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='checkpoint directory')
    parser.add_argument('--save_every', type=int, default=1000,
                       help='save frequency')
    args = parser.parse_args()

    # Read the training data
    inputFile = open("data/input.txt","rU")
    trainingData = inputFile.read()

    # Count vocab 
    counter = collections.Counter(trainingData)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    chars, _ = list(zip(*count_pairs))
    vocabSize = len(chars)
    print vocabSize
    vocab = dict(zip(chars, range(len(chars))))
    inputTensor = np.array(map(vocab.get, trainingData))

    numBatches = inputTensor.size / (batchSize * numSteps)

    print numBatches

    inputTensor = inputTensor[:numBatches * batchSize * numSteps]
    inData = inputTensor
    targetData = np.copy(inputTensor)
    targetData[:-1] = inData[1:]
    targetData[-1] = inData[0]
    inDataBatches = np.split(inData.reshape(batchSize, -1), numBatches, 1)
    targetDataBatches = np.split(targetData.reshape(batchSize, -1), numBatches, 1)
    
    lstmTrain(args)

def lstmTrain(args):
    data_loader = TextLoader('data', batchSize, numSteps)
    args.vocabSize = data_loader.vocab_size

    print args.vocabSize

    _lstmModel = LSTMModel(args)

    with tf.Session() as trainSess:

        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())

        for currEpoch in xrange(numEpochs):

            # For reading batches of training data.
	    currBatchPointer = 0

            # Set the learning rate. Decay after every epoch.
            trainSess.run(tf.assign(_lstmModel.learningRate, learningRate * decayRate ** e))
            state = _lstmModel.initialState.eval()

            for currBatch in xrange(numBatches):

                # Set input and target output data for current batch. 
                inData = inDataBatches[currBatchPointer]
                targetData = targetDataBatches[currBatchPointer]

                #print inData

                # We will feed the data to the session.
                inputFeed = {_lstmModel.inputData: x, _lstmModel.targetOutput: y, _lstmModel.initialState: state}

                trainLoss, state, _ = trainSess.run([_lstmModel.cost, _lstmModel.final_state, _lstmModel.trainStep], inputFeed)
                print "epoch".currEpoch
                print "trainingLoss".trainLoss

                # Save a checkpoint
                if currEpoch % 5 == 0:
                  checkpointPath = os.path.join(args.save_dir, 'lstmModel.ckpt')
                  saver.save(trainSess, checkpoint_path, global_step = currEpoch * numBatches + currBatch)
                  print "Saving checkpoint".format(checkpoint_path)

if __name__ == '__main__':
    main()
