from common import getTrainingFiles, getTrainingData, np, trainClassifier

trainingWav = getTrainingFiles(True, True)
trainingWav.sort()

MODE = 1

print("\n>> Extracting features")

X, y = [], []
for n, _file in enumerate(trainingWav):
  print(f"\nFile {n+1}/{len(trainingWav)} - {_file}")
  localX, localY = getTrainingData(_file, 0, True, "pitchTemp")
  y += localY
  for n, note in enumerate(localX):
    if MODE == 1: X.append([note["pitch"], note["harmonic"], note["harmonicAvg"], np.max(note["spectrogram"]), np.mean(note["spectrogram"])]); continue
    if n == 0: piece = ([-1]*12) + [note["pitch"], note["harmonic"], np.max(note["spectrogram"])]
    else: piece = localY[n-1].tolist() + [note["pitch"], note["harmonic"], np.max(note["spectrogram"])]
    X.append(piece)

for piece in X: print(piece)

print("\n>> Training")

classifier = trainClassifier(X, y, True, f'pitchClassifier{"" if MODE == 1 else "Prev"}')

print("\n>> Model saved\n")