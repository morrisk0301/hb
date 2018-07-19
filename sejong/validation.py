# -*- coding: utf-8 -*-
"""
Created on Sun May 13 19:38:07 2018

@author: 경인
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import sys
from commonfunction import *

#Preset
mfccsize = 13
mfcclength = 25
wavlength = 2.5
n_mels = 26
totalsize = mfccsize*mfcclength
loop = 0

if sys.argv[1] == '-f':
    loop = int(sys.argv[2])


#file이름, dataset이름 설정 
modelname = 'model/forcertify_'+str(loop)
dataset = 'data/forcertify_'+str(loop)+'_test.npz'

sound_data_t = np.load(dataset)
nparray_test = sound_data_t['X']/500
nparray_test_l = sound_data_t['Y']

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
Y = tf.placeholder(tf.float32, [None, n_class])
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
saver.restore(sess, modelname+'.ckpt')

#Validation
test_xs = nparray_test
test_xs.resize(test_size, 1, n_input)
test_ys = nparray_test_l

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

output = sess.run(tf.argmax(model, 1), feed_dict={X: test_xs, Y: test_ys, keep_prob: 1.0})
output_float = sess.run(model, feed_dict={X: test_xs, Y: test_ys, keep_prob: 1.0})
answer = sess.run(tf.argmax(Y, 1), feed_dict={X: test_xs, Y: test_ys, keep_prob: 1.0})


output_l = output.tolist()
output_log = np.exp(output_float).tolist()
output_float_l = output_float.tolist()
answer_l = answer.tolist()

column = ['Disease', 'Output', 'Disease(N/AN)', 'Output(N/AN)', 'AS', 
          'MR', 'MS', 'MVP', 'N']

index = []

for i in range(test_size):
    index.append(i)

right = 0
right_nan = 0
farcnt = 0
frrcnt = 0
data = pd.DataFrame(0, index = index, columns = column)
for i in range(test_size):
    n_an_a = 'N'
    n_an_o = 'N'
    if int(answer_l[i]) != 0 :
        n_an_a = 'AN'
    if int(output_l[i]) != 0 :
        n_an_o = 'AN'
    disease_excel = returnDisease(int(answer_l[i]))
    output_excel = returnDisease(int(output_l[i]))
    
    data.loc[i, 'Disease'] = disease_excel 
    data.loc[i, 'Output'] = output_excel
    data.loc[i, 'Disease(N/AN)'] = n_an_a
    data.loc[i, 'Output(N/AN)'] = n_an_o
    data.loc[i, 'Output'] = returnDisease(int(output_l[i]))
    data.loc[i, 'N'] = output_log[i][0]/sum(output_log[i])*100
    data.loc[i, 'AS'] = output_log[i][1]/sum(output_log[i])*100
    data.loc[i, 'MR'] = output_log[i][2]/sum(output_log[i])*100
    data.loc[i, 'MS'] = output_log[i][3]/sum(output_log[i])*100
    data.loc[i, 'MVP'] = output_log[i][4]/sum(output_log[i])*100
    
    if disease_excel == output_excel:
        right += 1
    if n_an_a == n_an_o:
        right_nan += 1
    if n_an_a == 'N' and n_an_o != 'N':
        farcnt += 1
    if n_an_a == 'AN' and n_an_o != 'AN':
        frrcnt += 1
    
data.to_excel('result/fold'+str(loop)+'_result.xlsx', index=True)

print("총 정확도: %.2lf%%" % (right*100/test_size))
print("N/AN 정확도: %.2lf%%" % (right_nan*100/test_size))
print("False Acceptance Rate: %.2lf%%" % (farcnt*100/36))
print("False Rejection Rate: %.2lf%%" % (frrcnt*100/144))