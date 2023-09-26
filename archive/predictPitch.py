import joblib
from addons.featuresSeparating import extractFeatures
from addons.sheet import extractData

classifier = joblib.load('pitchClassifier.pkl')

toPred = []
notes, _ = extractFeatures("media/test.wav")
for n, note in enumerate(notes):
  pitch = round(note["pitch"], 2)
  harmonic = note["harmonic"]
  spectrogram = note["spectrogram"]
  toPred.append([pitch, harmonic, spectrogram])

predictions = classifier.predict(toPred)

print(predictions)
print(len(predictions))
extractData(predictions, "test")