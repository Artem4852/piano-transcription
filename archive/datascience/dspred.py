import pickle
import numpy as np
np.set_printoptions(threshold=np.inf, suppress=True)
from addons.featuresSeparating import extractFeatures
from addons.sheet import extractData
from addons.changeNote import changeNote

harmonicOctaves = {
  "A": np.array([0.041730896, 0.03242255, 0.04036722, 0.04166268]),
  "D": np.array([0.0508472, 0.06189174]),
  "E": np.array([0.04602564, 0.05168208]),
  "G": np.array([0.045114204, 0.062146705])
}

harmonicNotes = {
  "CG": np.array([0.03857631, 0.0477372, 0.04182048, 0.06432732]),
  "AD": np.array([0.03242255, 0.04166268, 0.05212652, 0.03498119, 0.0508472]),
}

spectrogramLength = np.array([86, 43, 22, 11, 6])
restLength = np.array([86, 43, 22])

def approximateRest(length):
  rest_approximation = []
  for value in restLength:
    while length >= value:
      length -= value
      rest_approximation.append(restLength.tolist().index(value))
  return rest_approximation

with open("in.pkl", 'rb') as fileIn, open("out.pkl", 'rb') as fileOut:
  x = pickle.load(fileIn)
  y = pickle.load(fileOut)

notes, _ = extractFeatures("restTest.wav")

toPred = []
other = []
for n, note in enumerate(notes):
  toPred.append([note["pitch"], note["harmonic"], note["spectrogramLen"], note["spectrogramLowCol"]])

toPred = np.array(toPred)

print(toPred)

predictions = []
for piece in toPred:
  differences = np.abs(x[:, 0] - piece[0])
  closest_index = np.argmin(differences)
  closest_value = y[closest_index].copy()

  lenDifferences = np.abs(spectrogramLength - piece[2])
  lenClosestIndex = np.argmin(lenDifferences)

  if piece[3] != None and piece[3] > 5: 
    rests = approximateRest(piece[3])
  else: restLen = None

  # print(f"Before:   {closest_value}")

  # Correcting note
  # if (closest_value[6] or closest_value[10]) and (closest_value[0] or closest_value[1]):
  #   harmDifference = np.abs(harmonicNotes["CG"] - piece[1])
  #   harmClosestIndex = np.argmin(harmDifference)
  #   if harmClosestIndex in [0,1]: closest_value = changeNote(closest_value, note="C")
  #   else: closest_value = changeNote(closest_value, note="G")

  # elif (closest_value[4] or closest_value[7]) and (closest_value[0] or closest_value[1]):
  #   harmDifference = np.abs(harmonicNotes["AD"] - piece[1])
  #   harmClosestIndex = np.argmin(harmDifference)
  #   if harmClosestIndex in [0,2]: closest_value = changeNote(closest_value, note="A")
  #   else: closest_value = changeNote(closest_value, note="D")
  
  # # Correcting octave
  # if closest_value[4] and (closest_value[0] or closest_value[1]):
  #   harmDifference = np.abs(harmonicOctaves["A"] - piece[1])
  #   # print(f"A: {harmDifference}")
  #   harmClosestIndex = np.argmin(harmDifference)
  #   if harmClosestIndex in [0,1]: closest_value = changeNote(closest_value, octave=3)
  #   else: closest_value = changeNote(closest_value, octave=4)
  
  # elif closest_value[7] and (closest_value[1] or closest_value[2]):
  #   harmDifference = np.abs(harmonicOctaves["D"] - piece[1])
  #   harmClosestIndex = np.argmin(harmDifference)
  #   if harmClosestIndex == 0: closest_value = changeNote(closest_value, octave=4)
  #   else: closest_value = changeNote(closest_value, octave=5)
  
  # elif closest_value[8] and (closest_value[0] or closest_value[1]):
  #   harmDifference = np.abs(harmonicOctaves["E"] - piece[1])
  #   harmClosestIndex = np.argmin(harmDifference)
  #   if harmClosestIndex == 0: closest_value = changeNote(closest_value, octave=3)
  #   else: closest_value = changeNote(closest_value, octave=4)

  # elif closest_value[10] and (closest_value[0] or closest_value[1]):
  #   harmDifference = np.abs(harmonicOctaves["G"] - piece[1])
  #   harmClosestIndex = np.argmin(harmDifference)
  #   if harmClosestIndex == 0: closest_value = changeNote(closest_value, octave=3)
  #   else: closest_value = changeNote(closest_value, octave=4)
  
  # print(f"After:    {closest_value}\n")

  predictions.append([closest_value, lenClosestIndex, rests])

print(len(predictions))
extractData(predictions, "test")