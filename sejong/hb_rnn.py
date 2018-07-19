# 머신러닝 학습의 Hello World 와 같은 MNIST(손글씨 숫자 인식) 문제를 신경망으로 풀어봅니다.

import tensorflow as tf
import numpy as np
import sys
from commonfunction import *


tf.reset_default_graph()

loop = 0

if sys.argv[1] == '-f':
    loop = int(sys.argv[2])
    
dataset = 'data/forcertify_'+str(loop)+'_train.npz'
dataset_t = 'data/forcertify_'+str(loop)+'_test.npz'
modelname = 'model/forcertify_'+str(loop)

sound_data = np.load(dataset)
sound_data_t = np.load(dataset_t)

nparray = sound_data['X']/500
nparray_l = sound_data['Y']
nparray_test = sound_data_t['X']/500
nparray_test_l = sound_data_t['Y']

print('Traning set data size is', len(nparray))
print('Traning set label size is', len(nparray_l))
print('Test set data size is', len(nparray_test))
print('Test set label size is', len(nparray_test_l))

learning_rate = 0.0001
total_epoch = 500
batch_size = 1
test_size = len(nparray_test)

total_size, n_input = nparray.shape
n_step = 1
n_hidden = n_input
n_class = 5

#########

# 신경망 모델 구성

######

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


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


#########

# 신경망 모델 학습

######

sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(len(nparray)/batch_size)

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

cost_history = np.empty(shape=[1],dtype=float)

acc = 0

output = []
output_float = []
answer = []

saver = tf.train.Saver()
for epoch in range(total_epoch):
    start = 0    
    total_cost = 0
    for i in range(total_batch):
        end = (i+1)*batch_size
        
        array_x = nparray[start:end]
        array_x.resize(batch_size, 1, n_input)
        array_y = nparray_l[start:end]
        
        
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: array_x, Y: array_y, keep_prob: 0.5})

        total_cost += cost_val
        
        start = end

    if epoch %5 == 0:

        test_batch_size = len(nparray_test)

        test_xs = nparray_test
        test_xs.resize(test_size, 1, n_input)
        test_ys = nparray_test_l

        acc = sess.run(accuracy, feed_dict={X: test_xs, Y: test_ys, keep_prob: 1.0})

        print('정확도:', acc)

    cost_history = np.append(cost_history, total_cost / total_batch)
    print('Epoch:', '%04d' % epoch,

          'Avg. loss =', '{:.20f}'.format(total_cost / total_batch))
    
    if epoch == total_epoch-1:
        acc = sess.run(accuracy, feed_dict={X: test_xs, Y: test_ys, keep_prob: 1.0})
        print('정확도:', acc)
        
        output = sess.run(tf.argmax(model, 1), feed_dict={X: test_xs, Y: test_ys, keep_prob: 1.0})
        output_float = sess.run(model, feed_dict={X: test_xs, Y: test_ys, keep_prob: 1.0})
        answer = sess.run(tf.argmax(Y, 1), feed_dict={X: test_xs, Y: test_ys, keep_prob: 1.0})
        saver.save(sess, modelname+'.ckpt')
        print('Model이 저장되었습니다')
        print('학습이 완료되었습니')