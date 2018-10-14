import tensorflow as tf

# Define architecture
def model(X, w, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(
                     tf.nn.conv2d(
                     X, w,
                     strides=[1, 1, 1, 1],
                     padding='SAME'
                     )
                     + b1
                     )
    l1 = tf.nn.max_pool(
                        l1a,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME'
    )
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l3a = tf.nn.relu(
                     tf.nn.conv2d(
                     l1, w3,
                     strides=[1, 1, 1, 1],
                     padding='SAME'
                     )
                     + b3
                     )

    l3 = tf.nn.max_pool(
                        l3a,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME'
    )
    # Reshaping for dense layer
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w4) + b4)
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o) + b5
    return pyx

tf.reset_default_graph()

# Define variables
init_op = tf.global_variables_initializer()

X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 10])

w = tf.get_variable("w", shape=[4, 4, 1, 16],
                            initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable(name="b1", shape=[16],
                            initializer=tf.zeros_initializer())
w3 = tf.get_variable("w3", shape=[4, 4, 16, 32],
                            initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.get_variable(name="b3", shape=[32],
                            initializer=tf.zeros_initializer())
w4 = tf.get_variable("w4", shape=[32 * 7 * 7, 625],
                            initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.get_variable(name="b4", shape=[625],
                            initializer=tf.zeros_initializer())
w_o = tf.get_variable("w_o", shape=[625, 10],
                            initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.get_variable(name="b5", shape=[10],
                            initializer=tf.zeros_initializer())
# Dropout rate
p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

py_x = model(X, w, w3, w4, w_o, p_keep_conv, p_keep_hidden)

reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
reg_constant = 0.01

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y) + reg_constant * sum(reg_losses))

train_op = tf.train.RMSPropOptimizer(0.0001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

#Training
train_acc = []
val_acc = []
test_acc = []
train_loss = []
val_loss = []
test_loss = []
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    
    # Training iterations
    for i in range(256):
        # Mini-batch
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX)+1, batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end],
                                          Y: trY[start:end],
                                          p_keep_conv: 0.8,
                                          p_keep_hidden: 0.5})
        # Comparing labels with predicted values
        train_acc = np.mean(np.argmax(trY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: trX,
                                                         Y: trY,
                                                         p_keep_conv: 1.0,
                                                         p_keep_hidden: 1.0}))
        train_acc.append(train_acc)
        
        val_acc = np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX,
                                                         Y: teY,
                                                         p_keep_conv: 1.0,
                                                         p_keep_hidden: 1.0}))
        val_acc.append(val_acc)
        test_acc = np.mean(np.argmax(mnist.test.labels, axis=1) ==
                         sess.run(predict_op, feed_dict={X: mnist_test_images,
                                                         Y: mnist.test.labels,
                                                         p_keep_conv: 1.0,
                                                         p_keep_hidden: 1.0}))
        test_acc.append(test_acc)
        print('Step {0}. Train accuracy: {3}. Validation accuracy: {1}. \
Test accuracy: {2}.'.format(i, val_acc, test_acc, train_acc))
        
        _, loss_train = sess.run([predict_op, cost],
                              feed_dict={X: trX,
                                         Y: trY,
                                         p_keep_conv: 1.0,
                                         p_keep_hidden: 1.0})
        train_loss.append(loss_train)
        _, loss_val = sess.run([predict_op, cost],
                               feed_dict={X: teX,
                                         Y: teY,
                                         p_keep_conv: 1.0,
                                         p_keep_hidden: 1.0})
        val_loss.append(loss_val)
        _, loss_test = sess.run([predict_op, cost],
                              feed_dict={X: mnist_test_images,
                                         Y: mnist.test.labels,
                                         p_keep_conv: 1.0,
                                         p_keep_hidden: 1.0})
        test_loss.append(loss_test)
        print('Train loss: {0}. Validation loss: {1}. \
Test loss: {2}.'.format(loss_train, loss_val, loss_test))
    # Saving model
    all_saver = tf.train.Saver() 
    all_saver.save(sess, '/resources/data.chkp')

#Predicting
with tf.Session() as sess:
    # Restoring model
    saver = tf.train.Saver()
    saver.restore(sess, "./data.chkp")

    # Prediction
    pr = sess.run(predict_op, feed_dict={X: mnist_test_images,
                                         Y: mnist.test.labels,
                                         p_keep_conv: 1.0,
                                         p_keep_hidden: 1.0})

    print(np.mean(np.argmax(mnist.test.labels, axis=1) ==
                         sess.run(predict_op, feed_dict={X: mnist_test_images,
                                                         Y: mnist.test.labels,
                                                         p_keep_conv: 1.0,
                                                         p_keep_hidden: 1.0})))
