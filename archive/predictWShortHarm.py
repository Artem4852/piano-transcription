import joblib
import numpy as np
from addons.featuresSeparatingHarm import extractFeatures
from addons.sheet import extractData

pitchClassifier = joblib.load('models/pitchClassifier.pkl')
spectClassifier = joblib.load('models/spectShortClassifier.pkl')
restsClassifier = joblib.load('models/rest_spectClassifier.pkl')

toPredPitch = []
toPredSpect = []
toPredRests = []

print("\n>> Extracting features\n")

notes, _ = extractFeatures("media/test101.wav")
for n, note in enumerate(notes):
  toPredPitch.append([note["pitch"], note["harmonic"], np.max(note["spectrogram"])])
  toPredSpect.append([note["pitch"], note["harmonic"], note["spectrogramLen"], note["spectrogramLowCol"]])
  toPredRests.append([note["spectrogram"], note["harmonicShort"]])

print("\n>> Padding")
longestSpect = 1000
longestHarm = 100000

padded_X = []
for n, sp in enumerate(toPredRests):
  print(f"\033[1ANote {n+1}/{len(toPredRests)}")
  colsPadSpect = longestSpect - len(sp[0][0])
  padWidthSpect = ((0, 0), (0, colsPadSpect))

  paddedSpect = np.pad(sp[0], padWidthSpect, mode='constant', constant_values=-100)
  paddedSpect = paddedSpect.flatten().tolist()
  paddedSpect.insert(0, toPredPitch[n][0])
  paddedSpect.insert(1, toPredPitch[n][1])
  padded_X.append(paddedSpect + sp[1])

toPredRests = np.array(padded_X)

print("\n>> Predicting")

predictionsPitch = pitchClassifier.predict(toPredPitch)
predictionsSpect = spectClassifier.predict(toPredSpect)
predictionsRests = restsClassifier.predict(toPredRests)

# print(predictionsPitch)
# print(predictionsSpect)

# print(predictionsRests)

# print(len(predictionsPitch))

print("\n>> Extracting data\n")
extractData(predictionsPitch, predictionsSpect, predictionsRests, "short")