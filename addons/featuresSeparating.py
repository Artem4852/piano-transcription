import librosa, os
import numpy as np
import pandas as pd
import librosa.display
from pydub import AudioSegment

nameMain = __name__ == "__main__"

MARGINR = 25
MARGINL = 35

def separateNotes(filename, tempFolder):
    os.system(f"rm -rf {tempFolder}")
    os.makedirs(tempFolder, exist_ok=True)

    y, sr = librosa.load(filename)

    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    onset_times_ms = [int(time * 1000) for time in onset_times]

    song = AudioSegment.from_wav(filename)
    for i in range(len(onset_times)-1):
      trimmed_audio = song[onset_times_ms[i]-MARGINR:onset_times_ms[i+1]-MARGINL]
      trimmed_audio.export(f"{tempFolder}/note{i}.wav", format="wav")

    trimmed_audio = song[onset_times_ms[i+1]-MARGINL:]
    trimmed_audio.export(f"{tempFolder}/note{i+1}.wav", format="wav")

    return onset_times

def extractFeaturesSep(tempFolder):
    notes = []

    noteFiles = os.listdir(tempFolder)
    noteFiles.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    print()
    for n, noteFile in enumerate(noteFiles):
        print(f"\033[1ANote {n+1}/{len(noteFiles)}")
        y, sr = librosa.load(f"{tempFolder}/"+noteFile)

        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        log_spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)

        frequencies, magnitudes = librosa.piptrack(y=y, sr=sr)
        estimated_pitch = frequencies[np.argmax(magnitudes, axis=0)]

        harmonic, percussive = librosa.effects.hpss(y)

        note_frames = log_spectrogram

        mask = note_frames <= -50
        col = np.where(mask.all(axis=0))[0]
        try: col = min(col)
        except: col = -1

        notes.append({
            'pitch': estimated_pitch[0] if nameMain else round(float(np.mean([el for el in estimated_pitch[0] if el != 0])), 2),
            'spectrogram': note_frames,
            "spectrogramLen": note_frames.shape[1],
            "spectrogramLowCol": col,

            "harmonic": harmonic.flatten() if nameMain else np.max(harmonic),
            "harmonicFull": harmonic,
            "harmonicAvg": np.mean(harmonic),
            "percussive": np.max(percussive),
            "pervussiveAvg": np.mean(percussive)
        })
    
    os.system(f"rm -rf {tempFolder}")
    return notes

def extractFeatures(filename, tempFolder="notesTemp"):
    print("Separating notes")
    times = separateNotes(filename, tempFolder)
    print("Notes separated")
    notes = extractFeaturesSep(tempFolder)
    print("Features extracted")
    return notes, times

if nameMain:
    prefix = "rests_"

    notes, times = extractFeatures("test.wav")

    os.makedirs(f"media/{prefix}pitches", exist_ok=True)
    os.makedirs(f"media/{prefix}spectrograms", exist_ok=True)
    os.makedirs(f"media/{prefix}harmonic", exist_ok=True)

    # os.makedirs(f"{prefix}percussive", exist_ok=True)
    for n, note in enumerate(notes):
        print(note["harmonicFull"].shape)
        time = round(times[n], 2)
        pitchAvg = round(float(np.mean([el for el in note["pitch"] if el != 0])), 2)
        spectMax = str(round(np.max(note['spectrogram']), 2))
        harmonicMax = str(round(np.max(note['harmonic']), 8))
        pd.DataFrame(note["spectrogram"]).to_csv(f"media/{prefix}spectrograms/{n}_{spectMax}_{time}.csv", index=False, columns=None)
        pd.DataFrame(note["pitch"]).to_csv(f"media/{prefix}pitches/{n}_{pitchAvg}_{time}.csv", index=False)
        pd.DataFrame(note["harmonicFull"]).to_csv(f"media/{prefix}harmonic/{n}_{harmonicMax}_{time}.csv", index=False, columns=None)

        # np.savetxt(f"media/{prefix}harmonic/{n}_{harmonicMax}_{time}.csv", note["harmonic"], delimiter=',')

        print(pitchAvg, note["spectrogramLen"], note["spectrogramLowCol"])