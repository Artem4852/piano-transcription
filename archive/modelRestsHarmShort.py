from common import saveData, loadData, padSpectrogram, getTrainingFiles, getTrainingData, trainClassifier

LOADEXISTINGFEATURES = False

trainingWav = getTrainingFiles(dotted=False, allpitches=False)
trainingWav.sort()

print("\n>> Extracting features\n")
X, additionalX, y = [], [], []
for n, _file in enumerate(trainingWav):
  if LOADEXISTINGFEATURES: continue
  print(f"File {n+1}/{len(trainingWav)} - {_file}")
  localX, localY = getTrainingData(_file, 2, True, "restShortTemp")
  for note in localX:
    X.append(note["spectrogram"])
    additionalX.append([note["pitch"], note["harmonic"], note["harmonicShort"]])
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
  padded_X.append([additionalX[n]["pitch"], additionalX[n]["harmonic"]] + padSpectrogram(sp) + additionalX[n]["harmonicShort"])

print("\n>> Training")
classifier = trainClassifier(padded_X, y, True, 'rest_spectClassifierShort')

print("\n>> Model saved\n")