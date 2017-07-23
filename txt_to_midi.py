import midi
import os
from random import randint

class NoteEvent():
    def __init__(self, time, val, on):
        self.time = time
        self.val = val
        self.on = on

names = os.listdir('data/classical/txt/')
name = names[randint(0, len(names)-1)]
print name
        
#with open('data/classical/txt/' + name, 'rb') as f:
with open('rnn.txt', 'rb') as f:
    song_txt = f.read()

print song_txt

pattern = midi.Pattern()
track = midi.Track()
pattern.append(track)

split_song = song_txt.split(' ')

events_to_sort = []
time = 0
tempo = 40
val = 64
for time_event in split_song:
    events = time_event.split('/')

    min_dur = 0
    for event in events:
        delta_val, dur = event.split(',')
        delta_val = int(delta_val)
        dur = int(dur)

        val += delta_val
        
        # dur is inverse of duration...
        if dur > min_dur:
            min_dur = dur

        events_to_sort.append(NoteEvent(time, val, True))
        events_to_sort.append(NoteEvent(time + 16 / float(dur) * tempo, val, False))

    time = time + 16 / float(min_dur) * tempo

sorted_events = sorted(events_to_sort, key=lambda x: x.time)

track.append(midi.NoteOnEvent(tick=0, velocity=70, pitch=sorted_events[0].val))

for idx in range(1,len(sorted_events)):
    evt = sorted_events[idx]
    prev_evt = sorted_events[idx-1]
    tick = int(evt.time - prev_evt.time)
    pitch = int(evt.val)

    if evt.on:
        velocity = 70
    else:
        velocity = 0

    mid_evt = midi.NoteOnEvent(tick=tick, velocity=velocity, pitch=pitch)
    track.append(mid_evt)

track.append(midi.EndOfTrackEvent(tick=1))

midi.write_midifile("rnn.mid", pattern)

