import numpy as np

def changeNote(before, octave=None, note=None):
  if octave != None:
    octaveArr = np.array([False, False, False, False])
    octaveArr[octave-3] = True
    before[:4] = octaveArr
  
  if note != None:
    noteArr = np.array([
      note=="A",
      note=="B",
      note=="C",
      note=="D",
      note=="E",
      note=="F",
      note=="G"
    ])
    before[4:11] = noteArr

  return before