# coding=utf-8
import tensorflow as tf
import numpy as np
import os
from numpy import genfromtxt
rng = np.random
feature_nd = genfromtxt('C:\\Users\\Administrator\\Desktop\\数学建模\\reset-data.csv',dtype=float, delimiter=',')
n_samples=np.shape(feature_nd)[0]
feature=feature_nd[:,:np.shape(feature_nd)[1]-1]
price=feature_nd[:,-1]
# Parameters
learning_rate = 0.00000002
training_epochs = 10000
batch_size = 5
display_step = 200

# Network Parameters
n_hidden_1 = 30  # 1st layer number of features
n_hidden_2 = 30  # 2nd layer number of features
n_hidden_3 = 40
n_hidden_4 = 30
n_input = 11
  # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float")


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # layer_1 = tf.nn.sigmoid(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # layer_2 = tf.nn.sigmoid(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)

    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    final=tf.nn.sigmoid(out_layer)
    return final


# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
    'out': tf.Variable(tf.random_normal([n_hidden_4, 1]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'out': tf.Variable(rng.randn())
}


# def multilayer_perceptron(x, weights, biases):
#     # Hidden layer with RELU activation
#     layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
#     layer_1 = tf.nn.relu(layer_1)
#     # layer_1 = tf.nn.sigmoid(layer_1)
#     # Hidden layer with RELU activation
#     layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
#     layer_2 = tf.nn.relu(layer_2)
#     # layer_2 = tf.nn.sigmoid(layer_2)
#     # Output layer with linear activation
#     out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
#     final=tf.nn.sigmoid(out_layer)
#     return final
#
#
# # Store layers weight & bias
# weights = {
#     'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
#     'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
#     'out': tf.Variable(tf.random_normal([n_hidden_2, 1]))
# }
# biases = {
#     'b1': tf.Variable(tf.random_normal([n_hidden_1])),
#     'b2': tf.Variable(tf.random_normal([n_hidden_2])),
#     'out': tf.Variable(rng.randn())
# }



# Construct model
# with tf.device('/gpu:0'):
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_sum(tf.pow(pred-y, 2))/(2*n_samples)

# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))


optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# changed by zzq
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# optimizer = tf.train.GradientDescentOptimizer(0.5)
# train = optimizer.minimize(loss)

# Launch the graph
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    for epoch in range(training_epochs):
        sess.run(init)
        avg_cost = 0.

        _, c = sess.run([optimizer, cost], feed_dict={x: feature,
                                                      y: price})
        avg_cost += c

        # avg_cost/=total_batch
        # # Display logs per epoch step
        if epoch % display_step == 0:
            # print("Epoch:", '%04d' % (epoch + 1), "cost=", \
            #       "{:.9f}".format(avg_cost))
            print("{:.9f}".format(avg_cost))
    print("Optimization Finished!")
    test_feat = genfromtxt('C:\\Users\\Administrator\\Desktop\\数学建模\\reset-data.csv', delimiter=',')
    print (test_feat.shape)
    test_=test_feat[:,:np.shape(test_feat)[1]-1]
    price_test=test_feat[:,-1]
    # sess.run(pred,feed_dict={x:test_feat})
    # index = tf.argmax(pred, 1)
    # print (index.eval(feed_dict={x:test_feat}))
    # 这样就可以打印出值了
    # out=file('/home/zzq/kaggle/house/test_file/test.txt','a+')
    # out.write((index.eval(feed_dict={x:test_feat})))
    # out.close()
    my_list = pred.eval(feed_dict={x: test_})
    my_list=np.reshape(my_list,[-1])
    error=np.sum(np.abs(my_list-price_test))
    error_all=(error*20.0/len(my_list))
    print ("the the test result:")
    print (error_all)
    # if os.path.isfile('/home/zzq/result/benz/result.csv'):
    #     os.remove('/home/zzq/result/benz/result.csv')
    # out = open('/home/zzq/result/benz/result.csv', 'a+')
    # id_pro = 'ID,y\n'
    # out.write(id_pro)
    # out.close()
    # with open('check','w') as ch:
    #     np.savetxt(ch,my_list,delimiter=',')
    # for i in range(len(my_list)):
    #     out = open('/home/zzq/result/benz/result.csv', 'a+')
    #     my_list[i] =(max_-min_)*my_list[i]+min_
    #     xian = str(int(time_tag[i])) + ',' + '%.8f' % my_list[i] + '\n'
    #     out.write(xian)
    #     out.close()