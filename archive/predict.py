from common import loadModel, np, getTrainingData, padSpectrogram, extractData

pitchClassifier = loadModel('pitchClassifier')
spectClassifier = loadModel('spectShortClassifier')
restsClassifier = loadModel('rest_spectClassifier')

toPredPitch, toPredSpect, additionalX = [], [], []

X = getTrainingData("test", mode=0, training=False)
for note in X:
  toPredPitch.append([note["pitch"], note["harmonic"], np.max(note["spectrogram"])])
  additionalX.append(note["harmonicFull"])
  toPredSpect.append(note["spectrogram"])

# longestSpect = max([sp.shape[1] for sp in toPred])
longestSpect = 1000
longestHarm = 80000

print()
toPredPaddedSpect = []
toPredPaddedRests = []
for n, sp in enumerate(toPredSpect):
  print(f"\033[1ANote {n+1}/{len(toPredSpect)}")
  colsPadHarm = longestHarm - len(additionalX[n])
  padWidthHarm = ((0, colsPadHarm))

  paddedHarm = np.pad(additionalX[n], padWidthHarm, mode='constant', constant_values=-100).tolist()
  spect = padSpectrogram(sp)
  toPredPaddedSpect.append(spect + paddedHarm)
  toPredPaddedRests.append([toPredPitch[n][0], toPredPitch[n][1]] + spect + paddedHarm)

predictionsPitch = pitchClassifier.predict(toPredPitch)
predictionsSpect = spectClassifier.predict(toPredPaddedSpect)
predictionsRests = restsClassifier.predict(toPredPaddedRests)

extractData(predictionsPitch, predictionsSpect, predictionsRests, "long")