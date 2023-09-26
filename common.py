import numpy as np
import os, joblib, pickle
from utils.files import removeExtension
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from addons.featuresSeparatingHarm import extractFeatures
from addons.sheet import labelData, extractData

# Variables

longestSpect = 1000
longestHarm = 100000

# Functions

def saveData(data, filename):
  os.makedirs("models/features", exist_ok=True)
  with open(filename, 'wb') as _file:
    pickle.dump(data, _file)

def loadData(filename):
  with open(filename, 'rb') as _file:
    return pickle.load(_file)
  
def saveModel(model, filename):
  joblib.dump(model, f"models/{filename}.pkl")

def loadModel(filename):
  return joblib.load(f"models/{filename}.pkl")

def padSpectrogram(spectrogram):
  colsPadSpect = longestSpect - len(spectrogram[0])
  padWidthSpect = ((0, 0), (0, colsPadSpect))
  paddedSpect = np.pad(spectrogram, padWidthSpect, mode='constant', constant_values=-100)
  paddedSpect = paddedSpect.flatten().tolist()
  return paddedSpect

def padHarmonic(harmonic):
  colsPadHarm = longestHarm - len(harmonic)
  padWidthHarm = ((0, colsPadHarm))
  paddedHarm = np.pad(harmonic, padWidthHarm, mode='constant', constant_values=-100).tolist()
  return paddedHarm

def getTrainingFiles(dotted=False, allpitches=False, spaces=False, combinations=False):
  return [removeExtension(f) for f in os.listdir("media/training/wav") if (not "allpitches2" in f or allpitches) and (not "dotted" in f.lower() or dotted) and (not "spaces" in f.lower() or spaces) and (not "combinations" in f.lower() or combinations)]

def getTrainingData(filename, mode=0, training=True, tempFolder="notesTemp"):
  X, y = [], []
  notes, _ = extractFeatures(f"media/{'training/wav/' if training else ''}{filename}.wav", tempFolder)
  if training: labeledDataPitch, labeledDataLength, labeledDataRests = labelData(f"media/training/mxl/{filename}.mxl")
  for n, note in enumerate(notes):
    X.append(note)
    if training: y.append(labeledDataPitch[n] if mode==0 else labeledDataLength[n] if mode==1 else labeledDataRests[n])
  if training: return X, y
  else: return X

def trainClassifier(X, y, save=True, filename='rest_spectClassifier'):
  X = np.array(X); y = np.array(y)
  base_classifier = DecisionTreeClassifier()
  classifier = MultiOutputClassifier(base_classifier)
  classifier.fit(X, y)
  if save: saveModel(classifier, filename)
  else: return classifier