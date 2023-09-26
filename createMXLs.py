import music21
from music21 import converter, stream
import numpy as np

octaves = [3, 4, 5, 6]
notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

allNotes = [note+str(octave) for octave in octaves for note in notes]

output_stream = stream.Stream()
for noteBefore in allNotes:
  for noteAfter in allNotes:
    print(noteBefore, noteAfter)
    note1 = music21.note.Note(noteBefore)
    note1.quarterLength = 1
    note2 = music21.note.Note(noteAfter)
    note2.quarterLength = 1
    output_stream.append(note1)
    output_stream.append(note2)

# for noteBefore in allNotes:
#   for noteAfter in allNotes:
#     print(noteBefore, noteAfter)
#     note1 = music21.note.Note(noteBefore)
#     note1.quarterLength = 0.25
#     note2 = music21.note.Note(noteAfter)
#     note2.quarterLength = 0.25
#     output_stream.append(note1)
#     output_stream.append(note2)
  
output_stream.write('mxl', 'allCombinationsQuarter.mxl')