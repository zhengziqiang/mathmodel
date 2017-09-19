#coding=utf-8
import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
feature_nd = np.genfromtxt('/home/scs4450/PycharmProjects/数学建模/3-reset.csv',dtype=float, delimiter=',')
n_samples=np.shape(feature_nd)[0]
feature=feature_nd[:,:np.shape(feature_nd)[1]]
feature=np.float32(feature)
price=feature_nd[:,-1]
price=np.float32(price)

# n_samples=np.shape(feature_nd)[0]
# feature1=feature_nd[:,2:5]
# feature2=feature_nd[:,8:11]
# feature=np.concatenate([feature1,feature2],axis=1)
# feature=np.float32(feature)


W = tf.Variable(tf.random_uniform([11, 1], -2.0, 2.0))

b = tf.Variable(tf.random_uniform([1], -2.0, 2.0))

# W1= tf.Variable(tf.random_uniform([11, 1], -2.0, 2.0))
# b1= tf.Variable(tf.random_uniform([1], -2.0, 2.0))
# W=tf.cast(W,tf.float32)
# b=tf.cast(b,tf.float32)
# feature=tf.cast(feature,tf.float32)

hypothesis=tf.matmul(feature,W)+b

# hypothesis_tmp=tf.matmul(feature,W)+b
# hypothesis=tf.matmul(hypothesis_tmp,W1)+b1

cost = tf.reduce_mean(tf.square(hypothesis - price))
# a = tf.Variable(0.01)
optimizer = tf.train.GradientDescentOptimizer(0.005)
train = optimizer.minimize(cost)
# train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)



test_all=np.genfromtxt('/home/scs4450/PycharmProjects/数学建模/3-reset.csv',dtype=float, delimiter=',')
test_feature=test_all[:,:np.shape(test_all)[1]]
test_feature=np.float32(test_feature)
# test_price=test_all[:,-1]
# test_price=np.float32(test_price)

# test_feature1=test_all[:,2:5]
# test_feature2=test_all[:,8:11]
# test_feature=np.concatenate([test_feature1,test_feature2],axis=1)
# test_feature=np.float32(test_feature)


test_pred=tf.matmul(test_feature,W)+b

# test_pred_tmp=tf.matmul(test_feature,W)+b
# test_pred=tf.matmul(test_pred_tmp,W1)+b1



init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
max_step=10000
weight=[[-0.78788918],
       [-0.2506994 ],
       [-0.35246801],
       [-0.10626937],
       [-0.12819545],
       [ 0.74754256],
       [ 0.08432924],
       [ 0.68700051],
       [ 0.17661905],
       [-0.45325163],
       [-0.04345581]]
bias=[0.4412801]
test = sess.run(test_pred,feed_dict={W:weight,b:bias})
test_list = np.reshape(test, [np.shape(test_feature)[0]])
print(test_list)
test_output = test_list * 20.0 + 69.0
# test_output = np.array(test_output)
test_output = np.reshape(test_output,[-1, 1])

with open("output3.csv","w") as foo:
    np.savetxt(foo, test_output, delimiter=',')


for step in range(max_step):
    sess.run(train)
    if step % 200 == 0:
        print (sess.run(cost))
               # , sess.run(cost), sess.run(W), )
    if step==max_step-1:
        test=sess.run(test_pred)
        test_list=np.reshape(test,[np.shape(test_feature)[0]])
        print(test_list)
        test_output=test_list*20.0+65.0
        test_output=np.arrar(test_output)
        test_output=np.reshape(test_output[-1,1])
        with open("output1.csv") as foo:
            np.savetxt(foo,test_output,delimiter=',')
        # print (test_list)
        # test_list+=np.abs(np.min(test_list))
        print(sess.run(W),sess.run(b))
        # error=np.sum(np.abs(test_list-test_price))
        # print (error*20.0/np.shape(test_feature)[0])

# import tensorflow as tf
#
# x_data = [[1., 0., 3., 0., 5.],
#            [0., 2., 0., 4., 0.]]
# y_data = [1, 2, 3, 4, 5]
#
# W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
# b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
#
# hypothesis = tf.matmul(W, x_data) + b
#
# cost = tf.reduce_mean(tf.square(hypothesis - y_data))
#
# a = tf.Variable(0.1)
# optimizer = tf.train.GradientDescentOptimizer(a)
# train = optimizer.minimize(cost)
#
# init = tf.global_variables_initializer()
#
# sess = tf.Session()
# sess.run(init)
#
# for step in range(2001):
#     sess.run(train)
#     if step % 20 == 0:
#         print (step, sess.run(cost), sess.run(W), sess.run(b))