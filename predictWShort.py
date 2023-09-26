from common import loadModel, np, getTrainingData, padSpectrogram, padHarmonic, extractData

MODE = 0

pitchClassifier = loadModel(f'pitchClassifier{"" if MODE == 0 else "Prev"}')
# pitchClassifier = loadModel(f'pitchClassifierCombinationsQES{"" if MODE == 0 else "Prev"}')
spectClassifier = loadModel('spectShortClassifier')
restsClassifier = loadModel('rest_spectClassifier')

toPredPitch, toPredSpect, toPredRests = [], [], []

print("\n>> Extracting features\n")

X = getTrainingData("test345", mode=0, training=False)
for note in X:
  toPredPitch.append([note["pitch"], note["harmonic"], note["harmonicAvg"], np.max(note["spectrogram"]), np.mean(note["spectrogram"])])
  toPredSpect.append([note["pitch"], note["harmonic"], len(note["harmonicFull"].flatten().tolist()), note["spectrogramLen"]])
  toPredRests.append([note["spectrogram"], note["harmonicFull"]])

print("\n>> Padding\n")

padded_X = []
for n, sp in enumerate(toPredRests):
  print(f"\033[1ANote {n+1}/{len(toPredRests)}")
  padded_X.append([toPredPitch[n][0], toPredPitch[n][1]] + padSpectrogram(sp[0]) + padHarmonic(sp[1]))

print("\n>> Predicting")

predictionsSpect = spectClassifier.predict(toPredSpect)
predictionsRests = restsClassifier.predict(np.array(padded_X))

predictionsPitch = [] if MODE == 1 else pitchClassifier.predict(toPredPitch)
for n, toPred in enumerate(toPredPitch):
  if MODE == 0: continue
  if n == 0:
    toPred = [-1]*12 + toPred
  else:
    toPred = predictionsPitch[-1].tolist() + toPred
  predictionsPitch.append(pitchClassifier.predict([toPred])[0])

print("\n>> Extracting data\n")
extractData(predictionsPitch, predictionsSpect, predictionsRests, "short")