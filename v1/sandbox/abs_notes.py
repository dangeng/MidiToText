import midi
import numpy as np

def add_hand(track, notes, note_times):
    # Keep track of ticks
    ticks = 0
    
    # Iterate through the events in the track
    for e in track:
        # Keep track of which tick we're on
        ticks += e.tick
        
        # If event is an NoteOnEvent and velocity is not 0, then add note
        # Vel = 0 means note ends
        if isinstance(e, midi.NoteOnEvent) and e.data[1] != 0:
            # Get note value
            #note = midi.NOTE_VALUE_MAP_FLAT[e.data[0]] # (note name)
            note = e.data[0]
            
            # Add time of event
            note_times.append(ticks)
            
            # Add notes to dictionary key
            if ticks in notes:
                notes[ticks].append(note)
            else:
                notes[ticks] = [note]
        else:
            pass

def extract(fname="alb_esp1.mid"):
    pattern = midi.read_midifile(fname)
    
    # Containers to store event times and events
    note_times = []
    notes = {}
    
    # Get tracks and add to containers
    for i in range(len(pattern)):
        add_hand(pattern[i], notes, note_times)

    # Get rid of duplicates and sort event_times into list
    note_times = sorted(list(set(note_times)))
    
    # Placeholder data
    raw_data = np.zeros(len(notes)*128)
    
    # Add note vectors (len 128) into data
    for i in range(len(notes)):
        # notes_played are the played notes at some time note_times[i]
        notes_played = notes[note_times[i]]
        
        #print(notes_played)
        for note in notes_played:
            raw_data[(i*128)+note] = 1
    
    data = np.zeros((len(notes)-5, 128*5))
    labels = np.zeros((len(notes)-5, 128))
    
    for i in range(len(notes)-5):
        data[i] = raw_data[i*128:(i+5)*128]
        labels[i] = raw_data[(i+5)*128:(i+6)*128]
    
    return data, labels
