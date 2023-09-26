import librosa, os
import numpy as np
from pydub import AudioSegment
os.makedirs("media/notes", exist_ok=True)

MARGINR = 25
MARGINL = 35

y, sr = librosa.load('media/test.wav')

frequencies, magnitudes = librosa.piptrack(y=y, sr=sr)
estimated_pitch = frequencies[np.argmax(magnitudes, axis=0)]

onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
onset_times = librosa.frames_to_time(onset_frames, sr=sr)

print(onset_times)

pitchTog = [round(float(np.mean([el for el in estimated_pitch[frame] if el != 0])), 2) for frame in onset_frames]
print(pitchTog)

onset_times_ms = [int(time * 1000) for time in onset_times]
# onset_times_ms.insert(0, 0)

song = AudioSegment.from_wav('media/test.wav')
for i in range(len(onset_times)-1):
  trimmed_audio = song[onset_times_ms[i]-MARGINR:onset_times_ms[i+1]-MARGINL]
  trimmed_audio.export(f"media/notes/note{i}.wav", format="wav")

trimmed_audio = song[onset_times_ms[i+1]-MARGINL:]
trimmed_audio.export(f"media/notes/note{i+1}.wav", format="wav")