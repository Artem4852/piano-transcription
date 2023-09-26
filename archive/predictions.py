import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd

np.set_printoptions(threshold=np.inf)

from utils.features import extractFeatures
from utils.sheet import extractData

model = load_model("pitch.model")

notes, _ = extractFeatures("test.wav")

xPitch = []
for n, note in enumerate(notes):
  xPitch.append(note["pitch"])

xPitch = np.array(xPitch)

print(xPitch)

predictions = model.predict(xPitch)

print(predictions)
print(len(predictions))

# pd.DataFrame(predictions.to_csv(f"pred.csv", index=False))

extractData(predictions, True)