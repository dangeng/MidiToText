'''
A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function
import numpy as np
import midi

np.set_printoptions(suppress=3)

# Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

data = np.loadtxt("data/" + str(0) + ".txt")
labels = np.loadtxt("labels/" + str(0) + ".txt")

def shuffle_unison(a,b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

def get_data(batch_size):
    song_index = np.random.randint(1)
    #data = np.loadtxt("data/" + str(song_index) + ".txt")
    #labels = np.loadtxt("labels/" + str(song_index) + ".txt")
    shuffle_unison(data, labels)
    return data[:batch_size], labels[:batch_size]

import tensorflow as tf

# Parameters
learning_rate = 0.005
training_epochs = 100000
batch_size = 1000
display_step = 1

# Network Parameters
n_hidden_1 = 512 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 640 # MNIST data input (img shape: 28*28)
n_classes = 128 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.tanh(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.tanh(layer_2)
    # Output layer with linear activation
    out_layer = tf.nn.softmax(tf.matmul(layer_2, weights['out']) + biases['out'])
    return out_layer

mean = 0
std = .1

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], mean, std)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], mean, std)),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes], mean, std))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1], mean, std)),
    'b2': tf.Variable(tf.random_normal([n_hidden_2], mean, std)),
    'out': tf.Variable(tf.random_normal([n_classes], mean, std))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        # total_batch = int(mnist.train.num_examples/batch_size)
        total_batch = 10
        # Loop over all batches
        for i in range(total_batch):
            # batch_x, batch_y = mnist.train.next_batch(batch_size)
            batch_x, batch_y = get_data(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

def test_net():
    data, labels = get_data(1)
    with tf.Session() as sess:
        sess.run(init)
        return sess.run(pred, {x: data}), labels

def get_next_note(data):
    with tf.Session() as sess:
        sess.run(init)
        return sess.run(pred, {x: data})

def compose_music(deeznuts):
    song, _ = get_data(1)
    with tf.Session() as sess:
        for i in range(deeznuts):
            prev = np.array([song[0][-640:]])
            song = np.array([np.append(song[0], get_next_note(prev)[0])])
            print(i)
    return song

def write_music(song, speed=100):
    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)
    for i in range(len(song[0])//128):
        for note in range(128):
            if song[0][i*128+note] > .5:
                track.append(midi.NoteOnEvent(tick=0, velocity=100, pitch=note))
        track.append(midi.ControlChangeEvent(tick=speed))
        for note in range(128):
            track.append(midi.NoteOnEvent(tick=0, velocity=0, pitch=note))
    midi.write_midifile("test.mid", pattern)
