import gzip
import cPickle

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h

def porcentaje(numero):
    return (2*numero)/100

#train_set, valid_set, test_set = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
f = gzip.open('mnist.pkl.gz', 'rb')
#set = cPickle.load(f)
#train_set, valid_set, test_set = np.split(f.sample(frac=1), [int(.6*len(f)), int(.8*len(f))])
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set

x_data = train_x.astype('f4')  # the samples are the four first rows of data
y_data = one_hot(train_y.astype(int), 10)  # the labels are in the last row. Then we encode them in one hot code

x_valid_data = valid_x.astype('f4')
y_valid_data = one_hot(valid_y.astype(int), 10)

x_test_data = test_x.astype('f4')
y_test_data = one_hot(test_y.astype(int), 10)

print "\nSome samples..."
for i in range(20):
    print x_data[i], " -> ", y_data[i]
print
xaxisgen = []
yaxisgen = []

def calcula(var):
    x = tf.placeholder("float", [None, 784])  # samples de 0 a 785
    y_ = tf.placeholder("float", [None, 10])  # labels de 0 a 9

    W1 = tf.Variable(np.float32(np.random.rand(784, var)) * 0.1)
    b1 = tf.Variable(np.float32(np.random.rand(var)) * 0.1)

    W2 = tf.Variable(np.float32(np.random.rand(var, 10)) * 0.1)
    b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

    h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
    # h = tf.matmul(x, W1) + b1  # Try this!
    y = tf.nn.softmax(tf.matmul(h, W2) + b2)

    loss = tf.reduce_sum(tf.square(y_ - y))
    #loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

    train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)
    i = 0
    print "----------------------"
    print "   Start training and validation...  "
    print "----------------------"

    batch_size = 20
    iter=0
    error2=100
    error1=1000
    xaxis = []
    yaxis= []
    while abs(error2-error1)/float(error1) > 0.01:
    #for epoch in xrange(500):
        for jj in xrange(len(x_data) / batch_size):
            batch_xs = x_data[jj * batch_size: jj * batch_size + batch_size]
            batch_ys = y_data[jj * batch_size: jj * batch_size + batch_size]
            sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

        error1 = error2
        error2 = sess.run(loss, feed_dict={x: x_valid_data, y_: y_valid_data})
        if iter>0:
            xaxis.append(iter)
            yaxis.append(error1)

        iter = iter + 1
        print "Epoch #:", iter, "Error: ", sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})
        result = sess.run(y, feed_dict={x: batch_xs})
        for b, r in zip(batch_ys, result):
            print b, "-->", r
        print "Iteraciones", iter
        #print "Error1:", error1
        #print "Error2:", error2
        #print error2-error1
        print "----------------------------------------------------------------------------------"

    print "----------------------"
    print "   Start test...  "
    print "----------------------"
    error_test = 0
    iterac=0
        #error_test = sess.run(loss, feed_dict={x: x_test_data, y_: y_test_data})
        #result = sess.run(y, feed_dict={x: x_test_data})
        #for b, r in zip(y_test_data, result):
        #    print b, "-->", r
    result = sess.run(y, feed_dict={x: x_test_data})
    acierto=0
    for b, r in zip(y_test_data, result):
        if np.argmax(b) == np.argmax(r):
            acierto+=1


    print "La precision es: ", acierto/float(len(x_test_data))
    print "----------------------------------------------------------------------------------"
    xaxisgen.append(var)
    yaxisgen.append(acierto/float(len(x_test_data)))
    plt.plot(xaxis,yaxis)
    value = str(var)
    plt.title("Evolucion de entrenamiento con "+value+" neuronas")
    plt.xlabel("Iteraciones")
    plt.ylabel("Error")
    plt.show()
calcula(0)
calcula(20)
calcula(40)
calcula(80)
calcula(120)
calcula(140)
calcula(200)
plt.plot(xaxisgen,yaxisgen)
plt.title("Precision con diferentes hiperparametros")
plt.xlabel("Neuronas")
plt.ylabel("Precision")
plt.show()
