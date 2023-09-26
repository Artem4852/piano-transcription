from common import getTrainingFiles, getTrainingData, trainClassifier, saveData, loadData

LOADEXISTINGFEATURES = False

trainingWav = getTrainingFiles(True, True, False)
trainingWav.sort()

print("\n>> Extracting features\n")

X, y = [], []
for n, _file in enumerate(trainingWav):
  if LOADEXISTINGFEATURES: continue
  print(f"File {n+1}/{len(trainingWav)} - {_file}")
  localX, localY = getTrainingData(_file, 1, True, "spectShortTemp")
  for note in localX:
    X.append([note["pitch"], note["harmonic"], len(note["harmonicFull"].flatten().tolist()), note["spectrogramLen"]])
  y += localY
  print()

if LOADEXISTINGFEATURES:
  print(">> Loading existing features")
  X = loadData("models/features/spectShort_X.pkl")
  y = loadData("models/features/spectShort_y.pkl")
else:
  print(">> Saving features")
  saveData(X, "models/features/spectShort_X.pkl")
  saveData(y, "models/features/spectShort_y.pkl")

print("\n>> Training")

classifier = trainClassifier(X, y, True, 'spectShortClassifier')

print("\n>> Model saved\n")