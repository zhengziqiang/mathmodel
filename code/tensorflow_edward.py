import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from edward.models import Normal
feature_nd = np.genfromtxt('C:\\Users\\Administrator\\Desktop\\数学建模\\mlp_regression_train.csv',dtype=float, delimiter=',')
n_samples=np.shape(feature_nd)[0]
feature=feature_nd[:,:np.shape(feature_nd)[1]-1]
feature=np.float32(feature)
price=feature_nd[:,-1]
price=np.float32(price)
X = tf.placeholder(tf.float32, [np.shape(feature)[0], 11])
w = Normal(loc=tf.zeros(11), scale=tf.ones(11))
b = Normal(loc=tf.zeros(1), scale=tf.ones(1))
y = Normal(loc=ed.dot(X, w) + b, scale=tf.ones(np.shape(feature)[0]))

qw = Normal(loc=tf.Variable(tf.random_normal([11])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([11]))))
qb = Normal(loc=tf.Variable(tf.random_normal([1])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))

inference = ed.KLqp({w: qw, b: qb}, data={X: feature, y: price})
inference.run(n_samples=11, n_iter=250)
print("debug")
test_all=np.genfromtxt('C:\\Users\\Administrator\\Desktop\\数学建模\\mlp_regression_val.csv',dtype=float, delimiter=',')
test_feature=test_all[:,:np.shape(test_all)[1]-1]
test_feature=np.float32(test_feature)

test_price=test_all[:,-1]
test_price=np.float32(test_price)
y_post = ed.copy(y, {w: qw, b: qb})
y_post = Normal(loc=ed.dot(X, qw) + qb, scale=tf.ones(np.shape(feature)[0]))
ed.evaluate('mean_squared_error', data={X: test_feature, y_post: test_price})