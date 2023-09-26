import music21
from music21 import converter, stream
import numpy as np

def labelData(filename):
  score = converter.parse(filename)
  score_info = []

  for note in score.flatten().notesAndRests:
    if note.isRest: 
      timing = float(note.offset)
      length = float(note.quarterLength)
      score_info.append((0, None, timing, length))
    elif note.isNote:
      pitch = note.pitch.nameWithOctave
      timing = float(note.offset)
      length = float(note.quarterLength)
      score_info.append((1, pitch, timing, length))

  labeled_data_pitch, labeled_data_spect, labeled_data_rests = [], [], []
  for n, (_type, pitch, _, length) in enumerate(score_info):
    if _type == 0: continue
    octave = int(pitch[-1])
    note = pitch[0]
    isdotted = length not in [0.25, 0.5, 1.0, 2.0, 4.0]

    totalRestLength = 0
    offset = 1
    while len(score_info) > n+offset:
      if score_info[n+offset][0] == 0: totalRestLength += score_info[offset][3]
      else: break
      offset += 1

    labeled_piece_pitch = [
      octave == 3,
      octave == 4,
      octave == 5,
      octave == 6,
      note == "A",
      note == "B",
      note == "C",
      note == "D",
      note == "E",
      note == "F",
      note == "G",
      "#" in pitch
    ]

    labeled_piece_spect = [
      length == 0.25 if not isdotted else length/3*2 == 0.25,
      length == 0.5 if not isdotted else length/3*2 == 0.5,
      length == 1.0 if not isdotted else length/3*2 == 1.0,
      length == 2.0 if not isdotted else length/3*2 == 2.0,
      length == 4.0,
      isdotted,
    ]

    labeled_piece_rests = [0]*33
    # print(int(totalRestLength/0.25))
    labeled_piece_rests[int(totalRestLength/0.25)] = 1

    labeled_data_pitch.append(np.array(labeled_piece_pitch))
    labeled_data_spect.append(np.array(labeled_piece_spect))
    labeled_data_rests.append(np.array(labeled_piece_rests))

  # for spect in labeled_data_spect: print(spect)

  return labeled_data_pitch, labeled_data_spect, labeled_data_rests

def extractData(pitchLabels, spectLabels, restLabels, name):
  # for label in spectLabels.tolist():
    # print(label)

  score_info = []

  for n, pitchLabel in enumerate(pitchLabels):
    octave = 3 + np.argmax(pitchLabel[:4])
    note = ["A", "B", "C", "D", "E", "F", "G"][np.argmax(pitchLabel[4:11])]
    sharp = "#" if int(pitchLabel[11]) == 1 else ""
    length = [0.25, 0.5, 1.0, 2.0, 4.0][np.argmax(spectLabels[n][:5])]
    # if np.array(spectLabels[n][:5]).tolist() == [0, 0, 0, 0, 0]: length = 1.0
    isdotted = int(spectLabels[n][5]) == 1
    if restLabels[n][0] != 1 and 1 in restLabels[n] and len(restLabels)-1 != n:
      restLength = (len(restLabels[n])-1-np.argmax(restLabels[n][::-1]))*0.25
      # print(restLabels[n])
      # print(len(restLabels[n])-1-np.argmax(restLabels[n][::-1]))
      # print(restLength)
    else:
      restLength = 0

    score_info.append((octave, note+sharp, length, isdotted, restLength))

  output_stream = stream.Stream()
  for octave, note, length, isdotted, restLength in score_info:
    new_note = music21.note.Note(f"{note}{octave}")
    # print(f"{note}{octave}", length, restLength)
    print(f"{note}{octave} {length} {isdotted} {restLength}")
    noteLength = length+length/2*isdotted
    new_note.duration.quarterLength = noteLength-restLength if noteLength-restLength > 0 else noteLength
    if isdotted: 
      new_note.duration.quarterLength = noteLength-restLength if noteLength-restLength > 0 else noteLength
      new_note.duration.dots = 1 if not new_note.duration.quarterLength in [0.25, 0.5, 1.0, 2.0, 4.0] else 0
    output_stream.append(new_note)
    # print(f"{note}{octave} {new_note.duration.quarterLength} {new_note.duration.dots == 1} {restLength}")
    
    if restLength != 0:
      new_rest = music21.note.Rest()
      new_rest.duration.quarterLength = restLength
      output_stream.append(new_rest)

  output_stream.write('mxl', fp=f'media/{name}.mxl')

if __name__ == "__main__":
  labelData("test.mxl")