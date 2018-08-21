import tensorflow as tf
#import theano
a = tf.constant(5.0)
b = tf.constant(4.0)
c = a*b
sess = tf.Session()
# print(sess.run(c))
# sess.close()
# with tf.Session() as sess:
#    output = sess.run([c])
#    print(output)
# to create in visuaization form
FIle_Writer = tf.summary.FileWriter(
    "/home/anjana/anjana/PYTHON/tensor/graph", sess.graph)
print(sess.run(c))
sess.close()
saver = tf.train.saver()
