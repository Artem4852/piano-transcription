from common import getTrainingFiles, getTrainingData, np, trainClassifier, padSpectrogram, saveData, padHarmonic, loadData, longestSpect

LOADEXISTINGFEATURES = True

trainingWav = getTrainingFiles(True, True)
trainingWav.sort()

print("\n>> Extracting features\n")
X, additionalX, y = [], [], []
for n, _file in enumerate(trainingWav):
  if LOADEXISTINGFEATURES: continue
  print(f"File {n+1}/{len(trainingWav)} - {_file}")
  localX, localY = getTrainingData(_file, 1, True, "spectTemp")
  for note in localX:
    X.append(note["spectrogram"])
    additionalX.append([note["pitch"], note["harmonic"], note["harmonicFull"]])
  y += localY
  print()

if LOADEXISTINGFEATURES:
  print(">> Loading existing features")
  X = loadData("models/features/spect_X.pkl")
  additionalX = loadData("models/features/spect_additionalX.pkl")
  y = loadData("models/features/spect_y.pkl")
else:
  print(">> Saving features")
  saveData(X, "models/features/spect_X.pkl")
  saveData(additionalX, "models/features/spect_additionalX.pkl")
  saveData(y, "models/features/spect_y.pkl")

print(">> Padding\n")

longestHarm = 128 * longestSpect

padded_X = []
for n, sp in enumerate(X):
  print(f"\033[1ANote {n+1}/{len(X)}")
  padded_X.append(padSpectrogram(sp) + padHarmonic(additionalX[n][2]))

print("\n>> Training")

classifier = trainClassifier(padded_X, y, True, 'spectClassifier')

print("\n>> Model saved\n")