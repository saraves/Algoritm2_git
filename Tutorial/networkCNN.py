# simple python script to train a 1-layer neural network to classify cifar10 images use the tensorflow library
import time
import numpy as np
import tensorflow as tf
import os
# class written to replicate input_data from tensorflow.examples.tutorials.mnist for CIFAR-10
import cifar10_read


# Batch normalization
def bnorm(x,is_training):
  shape = x.get_shape().as_list()
  gamma = tf.Variable(tf.ones(shape[len(shape)-1]))
  beta = tf.Variable(tf.zeros(shape[len(shape)-1]))
  pop_mean = tf.Variable(tf.zeros(shape[len(shape)-1]),trainable=False)
  pop_var = tf.Variable(tf.ones(shape[len(shape)-1]),trainable=False)

  batch_mean,batch_var = tf.nn.moments(x,list(range(len(shape)-1)))
  decay = 0.9*is_training[0] + 1*(1-is_training[0])
  train_mean = tf.assign(pop_mean,pop_mean*decay + batch_mean*(1-decay))
  train_var = tf.assign(pop_var,pop_var*decay + batch_var*(1-decay))
  with tf.control_dependencies([train_mean,train_var]):
    u_mean = batch_mean*is_training[0]+pop_mean*(1-is_training[0])
    u_var = batch_var*is_training[0] + pop_var*(1-is_training[0])
    next_x = tf.nn.batch_normalization(x,u_mean,u_var,beta,gamma,10e-08)
  return next_x

# location of the CIFAR-10 dataset
#CHANGE THIS PATH TO THE LOCATION OF THE CIFAR-10 dataset on your local machine
data_dir = '/Users/saravestergren/Documents/Skola/Kandidatjobb/Algoritm2_git/FindHair/Datasets/'

# read in the dataset
print('reading in the CIFAR10 dataset')
dataset = cifar10_read.read_data_sets(data_dir, one_hot=True, reshape=False, distort_train=True)
training = True
using_tensorboard = True
write_file = False
##################################################
# tuning parameters
learning_rate=0.002
beta1=0.9
beta2=0.999
epsilon=1e-08
use_locking=False
name='Adam'
n_iter = 100
nbatch = 2
eval_steps = 10
dropout = 0.75
##################################################
# PHASE 1  - ASSEMBLE THE GRAPH

# 1.1) define the placeholders for the input data and the ground truth labels

# x_input can handle an arbitrary number of input vectors of length input_dim = d
# y_  are the labels (each label is a length 10 one-hot encoding) of the inputs in x_input
# If x_input has shape [N, input_dim] then y_ will have shape [N, 10]
input_dim = 32*32*3    # d

x_input = tf.placeholder(tf.float32, shape = [None, 32,32,3])
y_ = tf.placeholder(tf.float32, shape = [None, 10])
pkeep = tf.placeholder(tf.float32)

is_training = tf.placeholder(tf.float32,shape = 1)
# weights and biast
Nf1 = 64
Nf2 = 64
M1 = 100
initializer_conv2d=tf.contrib.layers.xavier_initializer_conv2d(uniform = False)
initializer=tf.contrib.layers.xavier_initializer(uniform = False)

F1 = tf.Variable(initializer_conv2d([5,5,3,Nf1]))
bf1 = tf.Variable(tf.constant(.1,shape = [Nf1]))
F2 = tf.Variable(initializer_conv2d([5,5,Nf1,Nf2]))
bf2 = tf.Variable(tf.constant(.1,shape = [Nf2]))

W1 = tf.Variable(initializer([8*8*Nf2, M1]))
b1 = tf.Variable(tf.constant(0.1, shape=[M1]))
W2 = tf.Variable(initializer([M1,10]))
b2 = tf.Variable(tf.constant(0.1, shape=[10]))
# model with Batch Normalization on each layer
S1 = tf.nn.conv2d(x_input, F1, strides = [1,1,1,1], padding = "SAME")
S1 = tf.nn.relu(bnorm(S1,is_training))
Hf1 = tf.nn.max_pool(S1, ksize = [1,3,3,1],strides = [1,2,2,1],padding = "SAME")
S2 = tf.nn.conv2d(Hf1, F2, strides = [1,1,1,1], padding = "SAME")
S2 = tf.nn.relu(bnorm(S2,is_training))
Hf2 = tf.nn.max_pool(S2, ksize = [1,3,3,1],strides = [1,2,2,1],padding = "SAME")
H_flat = tf.reshape(Hf2, [-1,int(8*8*Nf2)])
print(H_flat.get_shape())
H2 = tf.matmul(H_flat,W1)
H2 = tf.nn.relu(bnorm(H2,is_training))
H2_dropout = tf.nn.dropout(H2,pkeep)
y = tf.matmul(H2_dropout, W2)

# 1.4) define the loss funtion
# cross entropy loss:
# Apply softmax to each output vector in y to give probabilities for each class then compare to the ground truth labels via the cross-entropy loss and then compute the average loss over all the input examples
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# 1.5) Define the optimizer used when training the network ie gradient descent or some variation.
# Use gradient descent with a learning rate of .01
global_step = tf.Variable(0,dtype = tf.int32,trainable=False,name='global_step')
train_step = tf.train.AdamOptimizer(learning_rate,beta1,beta2,epsilon,use_locking,name).minimize(cross_entropy,global_step=global_step)

# (optional) definiton of performance measures
# definition of accuracy, count the number of correct predictions where the predictions are made by choosing the class with highest score
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 1.6) Add an op to initialize the variables.
init = tf.global_variables_initializer()

##################################################
if write_file:
  myfile = open('/home/chunx/waspfinal3/resultsCNN.txt','a')
# If using TENSORBOARD
if using_tensorboard:
  # keep track of the loss and accuracy for the training set
  tf.summary.scalar('training loss', cross_entropy, collections=['training'])
  tf.summary.scalar('training accuracy', accuracy, collections=['training'])
  # merge the two quantities
  tsummary = tf.summary.merge_all('training')

  # keep track of the loss and accuracy for the validation set
  tf.summary.scalar('validation loss', cross_entropy, collections=['validation'])
  tf.summary.scalar('validation accuracy', accuracy, collections=['validation'])
  # merge the two quantities
  vsummary = tf.summary.merge_all('validation')

##################################################


##################################################
# PHASE 2  - PERFORM COMPUTATIONS ON THE GRAPH
time_start = time.clock()
saver = tf.train.Saver(max_to_keep = 0)

# 2.1) start a tensorflow session
if training:
  with tf.Session() as sess:
    savefile = open('waspfinal3/savors.txt','w')
  ##################################################
    ##################################################
    # 2.2)  Initialize the network's parameter variables
    # Run the "init" op (do this when training from a random initialization)
    sess.run(init)
    # 2.3) loop for the mini-batch training of the network's parameters
    for i in range(n_iter):

        # grab a random batch (size nbatch) of labelled training examples

        batch = dataset.train.next_batch(nbatch)

        # create a dictionary with the batch data
        # batch data will be fed to the placeholders for inputs "x_input" and labels "y_"
        batch_dict = {
            x_input: batch[0], # input data
            y_: batch[1], # corresponding label
            is_training:[1.0],
            pkeep:dropout
         }

        # run an update step of mini-batch by calling the "train_stsession ep" op
        # with the mini-batch data. The network's parameters will be updated after applying this operation
        sess.run(train_step, feed_dict=batch_dict)

        # periodically evaluate how well training is going
        if i % eval_steps == 0:
            save_path = saver.save(sess,'./checkpoints/pre_model',global_step = i)
            savefile.write(save_path + '\n')
            print(i)
            # compute the performance measures on the training set by
            # calling the "cross_entropy" loss and "accuracy" ops with the training data fed to the placeholders "x_input" and "y_"

            #tr = sess.run([cross_entropy, accuracy], feed_dict = {x_input:dataset.train.images[1:5000], y_: dataset.train.labels[1:5000]})

            # compute the performance measures on the validation set by
            # calling the "cross_entropy" loss and "accuracy" ops with the validation data fed to the placeholders "x_input" and "y_"

            #val = sess.run([cross_entropy, accuracy], feed_dict={x_input:dataset.validation.images, y_:dataset.validation.labels})

            #info = [i] + tr + val
            #print(info)

    savefile.close()
    # evaluate the accuracy of the final model on the test data
    test_acc = sess.run(accuracy, feed_dict={x_input: dataset.test.images, y_: dataset.test.labels,is_training : [0],pkeep:1.0})
    time_cost = time.clock()-time_start
    final_msg = "with learning rate:"+str(learning_rate)+", batch size:"+str(nbatch)+", iters:"+str(n_iter)+', the test accuracy is:' + str(test_acc)+"\ncomputional time cose:"+str(time_cost)+"s"
    print(final_msg)

##################################################
# restore model and compute train & test loss+accuracy
with tf.Session() as sess:
  # If using TENSORBOARD
  if using_tensorboard:
  # set up a file writer and directory to where it should write info +
  # attach the assembled graph
    summary_writer = tf.summary.FileWriter('networkCNN_do/results', sess.graph)
  savefile = open('waspfinal3/savors.txt')
  for save_path in savefile.readlines():
    saver = tf.train.import_meta_graph(save_path.replace("\n",'')+'.meta')
    saver.restore(sess, save_path.replace("\n",''))
    tc,ta,ts = sess.run([cross_entropy, accuracy,tsummary], feed_dict = {x_input:dataset.train.images[1:5000], y_: dataset.train.labels[1:5000],is_training:[1],pkeep: dropout})
    vc,va,vs = sess.run([cross_entropy, accuracy,vsummary], feed_dict={x_input:dataset.validation.images, y_:dataset.validation.labels,is_training:[0],pkeep :1.0})
    step = sess.run(global_step)
    dic = {'step':step,"train loss":tc,"train acc":ta,"valida loss":vc,"valida acc":vc}
    print(step,ta,va)
    if write_file:
      myfile.write(str(dict)+'\n')
    ##################################################
    # If using TENSORBOARD
    if using_tensorboard:
      # compute the summary statistics and write to file
      summary_str = sess.run(tsummary, feed_dict = {x_input:dataset.train.images, y_: dataset.train.labels})
      summary_writer.add_summary(ts, step)

      summary_str1 = sess.run(vsummary, feed_dict = {x_input:dataset.validation.images, y_: dataset.validation.labels})
      summary_writer.add_summary(vs, step)
      ##################################################
if write_file:
      myfile.write(final_msg+"\n")
      myfile.write("\n")
      myfile.close()
