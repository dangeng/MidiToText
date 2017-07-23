from subprocess import check_output
import os

txt = check_output(['cd ~/Documents/Programming/ML/torch-rnn && th sample.lua -checkpoint cv/checkpoint_62300.t7 -length 2000 -gpu -1'], shell=True)

txt = txt[txt.find(' ') + 1 : txt.rfind(' ')]

try:
    os.remove('rnn.txt')
except OSError:
    pass

with open('rnn.txt', 'wb') as f:
    f.write(txt)

check_output(['cd ~/Documents/Programming/midi-ml && python2.7 txt_to_midi.py'], shell=True)

os.system('timidity ~/Documents/Programming/midi-ml/rnn.mid')
