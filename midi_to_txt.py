import midi
import math
import operator
from fractions import gcd
import os

class NoteEvent():
    
    def __init__(self, time, val, dur):
        """ Initiializes the wrapper
        time:
            absolute time the note was played
        note:
            absolute note
        dur:
            absolute duration of the note
        """

        self.time = time
        self.val = val
        self.dur = dur

    def __str__(self):
        return '<NoteEvent: time: ' + str(self.time) + ' note: ' + str(self.note) + ' dur: ' + str(self.dur)

    def __repr__(self):
        return '<NoteEvent: time: ' + str(self.time) + ' note: ' + str(self.val) + ' dur: ' + str(self.dur) + '>'

def midi_to_txt(pattern):

    notes = []

    for track in pattern:
        time = 0
        for i in range(len(track)):
            evt = track[i]
            time += evt.tick

            if isinstance(evt, midi.events.NoteOnEvent) and evt.data[1] != 0:
                dur = 0
                for find_end_ctr in range(i+1, len(track)):
                    chk = track[find_end_ctr]
                    dur += chk.tick
                    if chk.data[0] == evt.data[0] and chk.data[1] == 0:
                        break

                if dur != 0:
                    notes.append(NoteEvent(time, evt.data[0], dur))

    notes = sorted(notes, key=lambda x: (x.time, x.val))

    gcd_dict = {}

    for i in range(int(len(notes)*.95)):
        tempo_gcd = gcd(notes[i].time, notes[i+1].time)
        if tempo_gcd in gcd_dict:
            gcd_dict[tempo_gcd] += 1
        else:
            gcd_dict[tempo_gcd] = 1

    gcd_dict = sorted(gcd_dict.items(), key=operator.itemgetter(1))
    tempo = float(min(gcd_dict[-5:], key=lambda x: x[0])[0])

    for note in notes:
        note.time = int(note.time/tempo)
        note.dur = note.dur/tempo

    dur_lst = [note.dur for note in notes]
    most_common_duration = max(set(dur_lst), key=dur_lst.count)

    for note in notes:
        note.dur = int(round(8 * (most_common_duration / note.dur)))
        if note.dur == 0:
            note.dur = 1

    song_txt = '0,' + str(notes[0].dur)
    prev_val = notes[0].val
    prev_time = notes[0].time

    for note in notes[1:]:
        if note.time == prev_time:
            song_txt += '/'
        else:
            song_txt += ' '

        song_txt += str(note.val - prev_val) + ',' + str(note.dur)

        prev_time = note.time
        prev_val = note.val

    return song_txt

#names = os.listdir('data/classical/all/')
#
#count = 0
#for name in names:
#    count += 1
#    print str(count) + '/' + str(len(names))
#    pattern = midi.read_midifile('data/classical/all/' + name)
#    song_txt = midi_to_txt(pattern)
#
#    with open('data/classical/txt/' + name.split('.')[0] + '.txt', 'wb') as f:
#        f.write(song_txt)

