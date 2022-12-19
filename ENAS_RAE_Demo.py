
# %%
from __future__ import division
import numpy as np
import math
# %%
from scipy import stats
from numpy import *

import tensorflow as tf
import os
import argparse
import os, sys, glob
import time
import math
import numpy as np
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import scipy.io as scio
import gc
import logging
CUDA_VISIBLE_DEVICES=0
import numpy as np
import sklearn.preprocessing
#np.seterr(divide='ignore', invalid='ignore')

import keras
from keras import backend as K
from keras.models import Sequential, Model
from tensorflow.contrib.keras import layers
from keras.layers import Input, LSTM, RepeatVector
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.optimizers import SGD, RMSprop, Adam
from keras import objectives
from keras import regularizers
from sklearn.linear_model import Lasso

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--single_gpu', default=True, action='store_false', 
                    help='use single GPU')
parser.add_argument('--gpu', type=int, default=0, help='GPU device to use')
parser.add_argument('--save', type=str,  default='EXP',
                    help='path to save the final model')
parser.add_argument('--continue_train', action='store_true',
                    help='continue train from a checkpoint')
args = parser.parse_args()
global subNetNum
global maxIter

# --- IMPORT DEPENDENCIES ------------------------------------------------------+
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        cudnn.enabled=True

args.save = 'Emotion_6'#'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
if not os.path.exists(args.save):
        os.mkdir(args.save)
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info('Args: {}'.format(args))
#-----------------------------------------------------------------------------------------------------------------
class DSRAE():


    def __init__(self ,visible_size ,layer_size ,hidden_size):

        self.input_dim = visible_size # self.inputs.shape[-1]
        self.timesteps = 176

        self.activity_regularizer = regularizers.l1( 1 *10e-7)
        self.kernel_regularizer = regularizers.l2( 1 *10e-4)
        self.lN = layer_size
        self.hN = np.zeros((self.lN +1) ,dtype=int)
        self.hN[0] = hidden_size
        print(self.hN)
        print(self.hN[0])

        a = 0.10
        a1 = 2.00
        a2 = 2.00
        temp1 = np.random.uniform(-1 ,1 ,1)
        temp2 = np.random.uniform(-1 ,1 ,1)
        variate = np.random.randint(-2 ,2 ,size=(self.lN ,1))
        while self.hN[0] < 10:
            self.hN[0] = self.hN[0] + 10
        print(self.hN)
        print(self.hN[0])
        '''encoder'''
        self.inputs = Input(shape=(self.timesteps, self.input_dim,))
        #         self.en_layer = self.inputs
        self.en_layer = Dense(self.hN[0], activation='tanh',
                              activity_regularizer=self.activity_regularizer,
                              kernel_regularizer=self.kernel_regularizer)(self.inputs)
        for i in range(1 ,self.lN +1):
            variate[ i -1] = a* variate[i - 1] + a1 * temp1 * (self.hN[i - 1]) + a2 * temp2 * (self.hN[i - 1])
            self.hN[i] = self.hN[i - 1] + variate[i - 1]
            self.hN[i] = abs(self.hN[i])

            #             if self.hN[i] < 0:
            #                 self.hN[i] = self.hN[i] + self.hN[i-1]
            #                 self.hN[i] = abs(self.hN[i])
            while self.hN[i] > self.hN[i - 1]:
                self.hN[i] = self.hN[i] - self.hN[i - 1]
                while self.hN[i] < 10:
                    self.hN[i] = self.hN[i] + 10
            while self.hN[i] <= 10:
                self.hN[i] = self.hN[i] + 10
                while self.hN[i] > self.hN[i - 1]:
                    self.hN[i] = self.hN[i] - self.hN[i - 1]
            while self.hN[i] > 200:
                self.hN[i] = self.hN[i] - 100
            print(self.hN[i])
            self.en_layer = LSTM(self.hN[i], return_sequences=True,
                                 activity_regularizer=self.activity_regularizer,
                                 kernel_regularizer=self.kernel_regularizer)(self.en_layer)

        self.encoder = Model(self.inputs, self.en_layer)
        self.encoder.summary()
        '''decoder'''
        self.inputs = Input(shape=(self.timesteps, self.hN[i],))
        self.de_layer = self.inputs
        for i in range(self.lN - 1, 0, -1):
            self.de_layer = LSTM(self.hN[i], return_sequences=True, activation='tanh')(self.de_layer)
        self.de_layer = LSTM(self.hN[0], return_sequences=True, activation='tanh')(self.de_layer)

        self.dense_layer = Dense(self.input_dim, activation='tanh')(self.de_layer)
        self.decoder = Model(self.inputs, self.dense_layer)
        self.decoder.summary()

        self.inputs = Input(shape=(self.timesteps, self.input_dim,))
        self.outputs = self.encoder(self.inputs)
        self.outputs = self.decoder(self.outputs)

        self.sequence_autoencoder = Model(self.inputs, self.outputs)

        self.errors = []
        
        
    def train(self, data):

        self.sequence_autoencoder.compile(optimizer='adam', loss='mse')
        self.sequence_autoencoder.summary()
        logging.info('-' * 89)
        logging.info('model summary '
                    .format( self.sequence_autoencoder.summary()))
        self.sequence_autoencoder.fit(data, data, epochs=10, batch_size=1)
        
        print(self.sequence_autoencoder.history.history['loss'])
        return self.sequence_autoencoder.history.history['loss'][-1]

        # raeup inspired by https://github.com/myme5261314/dbn_tf/blob/master/rbm_tf.py

    def predict_rae(self, data):
        y = np.zeros((np.shape(data)[0], np.shape(data)[1], self.hN[-1]), dtype=float)
        for i in range(0, np.shape(data)[0]):
            y[i * 1: (i + 1) * 1] = self.encoder.predict(data[i * 1: (i + 1) * 1])
        print(y)
        return y
    
    def get_structure(self):
        
        return self.hN


# %%


# --- Initialization ---------------------------------------------------------------------+
len_volume =176
sub_data = np.memmap('/home/qing/PycharmProjects/RAE/sub_Emotion.mymemmap', dtype='float32', mode='r+', shape=(len_volume * 791, 59421))
### zscore y#####

#sub_data = np.memmap('sub_WM.mymemmap', dtype='float32', mode='r+', shape=(len_volume * 791, 59421))

# Set the number of subnets(seeds)

subNetNum = 5
# Set the range of layers to search
layerNum = np.random.randint(1, 4, size=subNetNum)
# Set the range of units in each layer
unitNum = np.random.randint(100, 200, size=subNetNum)
# Set the initial mutation velocity to search
mutateVel = np.random.randint(-2, 2, size=(subNetNum, 2))
# Construct the NAS
structureNAS = np.zeros((subNetNum, 2))
structureNAS[:, 0] = layerNum
structureNAS[:, 1] = unitNum
# inputData = np.random.uniform(1,5,(186,2000))
data = np.expand_dims(sub_data, axis=1)
inputs = np.reshape(data, (791, len_volume, 59421))  # sub_data
inputs = inputs[0:700, :, :]
error = np.random.randint(-10, -1, size=(subNetNum, 1))
# Total iterative number

maxIter = 3
pBestStructure = structureNAS
pBestError = np.zeros(subNetNum)
gBestError = 0.0
gBestSeq = np.zeros(maxIter)
gBestStructure = np.zeros(2)
# errorOriginal = np.zeros(subNetNum)
error = np.zeros(subNetNum)
errorDefault = np.zeros(3)
errorSeq = np.zeros((subNetNum, maxIter))
w = 0.10
c1 = 2.00
c2 = 2.00

# --- Pre-Train and Calculate the Error------------------------------------------+
for ss in range(0, subNetNum):  #
    raeTemp = DSRAE(inputs.shape[-1], int(structureNAS[ss, 0]), int(structureNAS[ss, 1]))
    pBestError[ss] = raeTemp.train(inputs)
    raeTemp1 = raeTemp.predict_rae(inputs)
    print(pBestError[ss])
gBestStructure = structureNAS[np.argmin(pBestError),:]
gBestError = min(pBestError)

best_predict = raeTemp1

# --------------------------------------------------------------------------------+
# ----Main Loop (Search)----------------------------------------------------------+
gBestError = 2
#pBestError = 2
for it in range(0, maxIter):
    for i in range(0, subNetNum):
        rand1 = np.random.uniform(-1, 1, 1)
        rand2 = np.random.uniform(-1, 1, 1)
        mutateVel[i, :] = w * mutateVel[i, :] + c1 * rand1 * (pBestStructure[i, :] - structureNAS[i, :]) + c2 * rand2 * (gBestStructure - structureNAS[i, :])
        #structureNAS[i, :] = structureNAS[i, :] + mutateVel[i, :]
        structureNAS[i, :] = gBestStructure + mutateVel[i, :]
        print(structureNAS[i, :])
        if structureNAS[i, 0] == 0 :
            continue
        elif structureNAS[i, 0]>4:
            continue
        else:
            structureNAS[i, :] = abs(structureNAS[i, :])
            raeInput = DSRAE(inputs.shape[-1], int(structureNAS[i, 0]), int(structureNAS[i, 1]))


            error[i] = raeInput.train(inputs)
            print(gBestError)
            if math.isnan(error[i]):
                continue
            else:
                 if error[i] < gBestError:
                    gBestStructure = structureNAS[i, :]
                    gBestError = error[i]
                    rae_predict = raeInput.predict_rae(inputs)
                    best_predict = rae_predict
                    structure_list = raeInput.get_structure()
                    
                    print('aaaaaaaaaaaaaaaaaaaaaaaaa')
                 if pBestError[i] < gBestError:
                    gBestStructure = structureNAS[i,:]
                    structure_list = raeInput.get_structure()
                    rae_predict = raeInput.predict_rae(inputs)
                    best_predict = rae_predict
                    print(structure_list)
                
                 errorSeq[i, it - 1] = error[i]

    gBestSeq[it - 1] = gBestError
    

print('We have neural architecture search results here:')
print('NAS accurarcy')
print(gBestError)
print('NAS layer number')
print(int(gBestStructure[0]))
print('NAS unit number')
print(int(gBestStructure[1]))
print('NAS predict')
print(best_predict)
np.savetxt(args.save+'/ErrorSquence.txt', errorSeq)
np.savetxt(args.save+'/structure.txt', structure_list)
np.savetxt(args.save+'/Error.txt', pBestError)

import scipy.io as sio
sio.savemat(args.save+'/hidden_RNN.mat',{'hidden':best_predict})
#sio.savemat('WM_best_predict700total.mat', {'best_predict': best_predict})

if __name__ == "__ENAS_RAE__":
    main()
    
# --- END ----------------------------------------------------------------------+

# %%


