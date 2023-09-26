import joblib
import numpy as np
from addons.featuresSeparating import extractFeatures
from addons.sheet import extractData

classifier = joblib.load('spectClassifier.pkl')

toPred = []
notes, _ = extractFeatures("media/test.wav")
for n, note in enumerate(notes):
  spectrogram = note["spectrogram"]
  toPred.append(spectrogram)

# longestSpect = max([sp.shape[1] for sp in toPred])
longestSpect = 1000

toPredPadded = []
for sp in toPred:
  num_cols_to_pad = longestSpect - len(sp[0])
  pad_width = ((0, 0), (0, num_cols_to_pad))
  padded_sp = np.pad(sp, pad_width, mode='constant', constant_values=-100)
  toPredPadded.append(padded_sp.flatten())

toPred = np.array(toPredPadded)

predictions = classifier.predict(toPred)

print(predictions)
print(len(predictions))
extractData(predictions, "test")