from common import getTrainingFiles, getTrainingData, trainClassifier, padSpectrogram, padHarmonic, saveData, loadData

LOADEXISTINGFEATURES = True

trainingWav = getTrainingFiles()
trainingWav.sort()

print("\n>> Extracting features\n")
X, additionalX, y = [], [], []
for n, _file in enumerate(trainingWav):
  if LOADEXISTINGFEATURES: continue
  print(f"File {n+1}/{len(trainingWav)} - {_file}")
  localX, localY = getTrainingData(_file, 2, True, "restTemp")
  for note in localX:
    X.append(note["spectrogram"])
    additionalX.append([note["pitch"], note["harmonic"], note["harmonicFull"]])
  y += localY
  print()

if LOADEXISTINGFEATURES:
  print(">> Loading existing features")
  X = loadData("models/rest_X.pkl")
  additionalX = loadData("models/rest_additionalX.pkl")
  y = loadData("models/rest_y.pkl")
else:
  print(">> Saving features")
  saveData(X, "models/rest_X.pkl")
  saveData(additionalX, "models/rest_additionalX.pkl")
  saveData(y, "models/rest_y.pkl")

print("\n>> Padding\n")

padded_X = []
for n, sp in enumerate(X):
  print(f"\033[1ANote {n+1}/{len(X)}")
  padded_X.append([additionalX[n][0], additionalX[n][1]] + padSpectrogram(sp) + padHarmonic(additionalX[n][2]))

print("\n>> Training")

classifier = trainClassifier(padded_X, y, True, 'rest_spectClassifier')

print("\n>> Model saved\n")