import numpy as np

data = np.loadtxt("data/0.txt")
labels = np.loadtxt("labels/0.txt")

for i in range(1,929):
    data = np.vstack((data, np.loadtxt("data/" + str(i) + ".txt")))
    labels = np.vstack((labels, np.loadtxt("labels/" + str(i) + ".txt")))
    print i

np.savetxt("data.txt", data)
np.savetxt("labels.txt", labels)
