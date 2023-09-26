from utils.features import extractFeatures as extractFeaturesHarm
from utils.features import extractFeatures

import numpy as np
import pandas as pd

import librosa, os, pickle, random

notes, _ = extractFeatures("test.wav")
notesHarm, _ = extractFeaturesHarm("test.wav")

notes = np.array([note["spectrogram"] for note in notes])
notesHarm = np.array([note["spectrogram"] for note in notesHarm])

print(notes - notesHarm)