import abs_notes as extractor
import fnmatch
import os
import numpy as np

# filename of all midis
midi_file_names = []

# get filename of all midis
for root, dirnames, filenames in os.walk(os.getcwd()):
    for filename in fnmatch.filter(filenames, '*.mid'):
        midi_file_names.append(os.path.join(root, filename))

# get data and labels for all midis and save them to _data as txt
for i in range(len(midi_file_names)):
    d, l = extractor.extract(midi_file_names[i])
    np.savetxt("_data/data/" + str(i) + ".txt", d)
    np.savetxt("_data/labels/" + str(i) + ".txt", l)
    print i
