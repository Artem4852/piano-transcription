import numpy as np
from utils.files import removeExtension
import os, random, pickle

from addons.featuresSeparating import extractFeatures
from addons.sheet import labelData

trainingWav = [removeExtension(f) for f in os.listdir("training/wav")]
trainingWav.sort()

print("Extracting features")

x, y = [], []
for _file in trainingWav:
  notes, _ = extractFeatures(f"training/wav/{_file}.wav")
  labeledDataPitch, _ = labelData(f"training/mxl/{_file}.mxl")
  for n, note in enumerate(notes):
    pitch = round(note["pitch"], 2)
    harmonic = note["harmonic"]
    x.append([pitch, harmonic])
    y.append(labeledDataPitch[n])

x = np.array(x)
y = np.array(y)

print("Saving")

# os.makedirs("harm", exist_ok=True)
with open("in.pkl", 'wb') as fileIn, open("out.pkl", 'wb') as fileOut:
  pickle.dump(x, fileIn)
  pickle.dump(y, fileOut)

print("Done")