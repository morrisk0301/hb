# -*- coding: utf-8 -*-
"""
Created on Sun May 13 19:38:07 2018

@author: 경인
"""
import tensorflow as tf
import numpy as np
import librosa

#Preset
mfccsize = 13
mfcclength = 25
wavlength = 2.5
n_mels = 26
totalsize = mfccsize*mfcclength

#file이름, model이름 설정 
filename = 'sounds/sample_N.wav'
modelname = 'samplemodel'

#Wav file read
X, sample_rate = librosa.load(filename, sr=None)

#Preprocessing
duration = librosa.get_duration(y=X, sr=sample_rate)
duration_rate = 1/(wavlength/duration)
X_2 = librosa.effects.time_stretch(X, duration_rate)        

#MFCC extraction
mfcc = librosa.feature.mfcc(y=X_2, sr=sample_rate, n_mfcc=13, n_mels=n_mels, n_fft=800, hop_length=800)    

#Deeplearning appropriate input
mfcc = np.resize(mfcc, (mfccsize, mfcclength))
mfcc = np.resize(mfcc, (1, totalsize))
nparray_test = mfcc/500

#--------------------------------------------------------

tf.reset_default_graph()

#Tensorflow preset
learning_rate = 0.0001
test_size = len(nparray_test)

n_input = totalsize
n_step = 1
n_hidden = n_input
n_class = 5

X = tf.placeholder(tf.float32, [None, n_step, n_input])
W = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
outputs = tf.transpose(outputs, [1, 0, 2])
outputs = outputs[-1]
model = tf.matmul(outputs, W) + b

keep_prob = tf.placeholder(tf.float32)
h_3_drop = tf.nn.dropout(model, keep_prob)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#Load Model
saver = tf.train.Saver()
saver.restore(sess, 'model/'+modelname+'.ckpt')

#Validation
test_xs = nparray_test
test_xs.resize(test_size, 1, n_input)

validation = sess.run(model, feed_dict={X: test_xs, keep_prob: 1.0})
validation = np.exp(validation)

print("N percentage is", validation[0][0]/sum(validation[0])*100)
print("AN percentage is", (1-validation[0][0]/sum(validation[0]))*100)