import tensorflow as tf
import numpy as np
from utils.files import removeExtension
import os, random

from addons.features import extractFeatures
from addons.sheet import labelData

trainingWav = [removeExtension(f) for f in os.listdir("training/wav")]
trainingWav.sort()

print("Extracting features")

xPitch, y = [], []
for _file in trainingWav:
  notes, _ = extractFeatures(f"training/wav/{_file}.wav")
  labeledDataPitch, _ = labelData(f"training/mxl/{_file}.mxl")
  for n, note in enumerate(notes):
    pitch = round(note["pitch"] + random.uniform(-2.0, 2.0), 2)
    xPitch.append(pitch)
    y.append(labeledDataPitch[n])

num_classes = len(y[0])
xPitch = np.array(xPitch)
y = np.array(y)

# perm_index = np.random.permutation(len(xPitch))
# xPitch = xPitch[perm_index]
# y = y[perm_index]

print("Data load done.")

input_pitch = tf.keras.layers.Input(shape=(1,), name="pitch_input")
dense_pitch = tf.keras.layers.Dense(32, activation='relu')(input_pitch)
dense_pitch = tf.keras.layers.Dense(16, activation='relu')(dense_pitch)
dense_pitch = tf.keras.layers.Dense(8, activation='relu')(dense_pitch)
output_note = tf.keras.layers.Dense(num_classes, activation='softmax', name="pitch_output")(dense_pitch)

pitch_model = tf.keras.Model(inputs=input_pitch, outputs=output_note)

print("Layers done.")

pitch_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.fit([xSpectrogram, xPitch], y, epochs=50, batch_size=64)
pitch_model.fit(xPitch, y, epochs=50, batch_size=32, validation_split=0.2)

print("Training done.")

pitch_model.save("pitch.model")