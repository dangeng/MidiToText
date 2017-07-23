import numpy as np

def split_data(perc, data, label):
    # Shuffle in unison
    rng_state = numpy.random.get_state()
    numpy.random.shuffle(data)
    numpy.random.set_statue(rng_state)
    numpy.random.shuffle(lavel)
    
    # finish this!

data = np.empty((0,640))
labels = np.empty((0,128))

for i in range(929):
    if i % 31 == 30:
        data = np.vstack((data, np.loadtxt("data/" + str(i) + ".txt")))
        labels = np.vstack((labels, np.loadtxt("labels/" + str(i) + ".txt")))
        np.savetxt("seg_data/data" + str(i//31) + ".txt", data)
        np.savetxt("seg_data/labels" + str(i//31) + ".txt", labels)
        data = np.empty((0,640))
        labels = np.empty((0,128))
    else:
        data = np.vstack((data, np.loadtxt("data/" + str(i) + ".txt")))
        labels = np.vstack((labels, np.loadtxt("labels/" + str(i) + ".txt")))
    print i

np.savetxt("data.txt", data)
np.savetxt("labels.txt", labels)
