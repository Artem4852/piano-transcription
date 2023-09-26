import librosa, os
import numpy as np
import pandas as pd
import librosa.display
import matplotlib.pyplot as plt

nameMain = __name__ == "__main__"

MARGINR = 25
MARGINL = 35

def extractFeatures(filename):
    y, sr = librosa.load(filename)

    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    log_spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)

    frequencies, magnitudes = librosa.piptrack(y=y, sr=sr)
    estimated_pitch = frequencies[np.argmax(magnitudes, axis=0)]

    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    print(y.shape, sr)
    print(onset_frames)

    harmonic, percussive = librosa.effects.hpss(y)

    harm_onset_frames = [int(frame * (harmonic.shape[0]/estimated_pitch.shape[0])) for frame in onset_frames]

    # plt.figure(figsize=(10, 6))
    # librosa.display.waveshow(harmonic, sr=sr, alpha=0.5, label='Harmonic')
    # librosa.display.waveshow(percussive, sr=sr, color='r', alpha=0.5, label='Percussive')
    # plt.title('Harmonic and Percussive Components')
    # plt.legend(loc='upper right')
    # plt.show()
    # exit()

    notes = []
    for i in range(len(onset_frames)):
        start_frame = onset_frames[i]
        end_frame = onset_frames[i + 1] if i < len(onset_frames) - 1 else -1

        harm_start_frame = harm_onset_frames[i]
        harm_end_frame = harm_onset_frames[i + 1] if i < len(harm_onset_frames) - 1 else -1

        # print(harm_end_frame, harm_end_frame)
        # print(percussive[harm_end_frame:harm_end_frame])

        note_frames = log_spectrogram[:, start_frame:end_frame]
        notes.append({
            'start_frame': start_frame,
            'end_frame': end_frame,
            'pitch': estimated_pitch[start_frame] if nameMain else round(float(np.mean([el for el in estimated_pitch[start_frame] if el != 0])), 2),
            'spectrogram': note_frames if nameMain else round(np.max(note_frames), 2),
            "harmonic": harmonic[harm_start_frame:harm_end_frame] if nameMain else np.max(harmonic[harm_start_frame:harm_end_frame]),
            # "percussive": percussive[harm_end_frame:harm_end_frame]
        })

    return notes, onset_times

if nameMain:
    prefix = "test_"

    notes, times = extractFeatures("test.wav")

    os.makedirs(f"{prefix}pitches", exist_ok=True)
    os.makedirs(f"{prefix}spectrograms", exist_ok=True)
    os.makedirs(f"{prefix}harmonic", exist_ok=True)
    # os.makedirs(f"{prefix}percussive", exist_ok=True)
    for n, note in enumerate(notes):
        time = round(times[n], 2)
        pitchAvg = round(float(np.mean([el for el in note["pitch"] if el != 0])), 2)
        spectMax = str(round(np.max(note['spectrogram']), 2))
        harmonicMax = str(round(np.max(note['harmonic']), 8))
        pd.DataFrame(note["spectrogram"]).to_csv(f"{prefix}spectrograms/{n}_{spectMax}_{time}.csv", index=False, columns=None)
        pd.DataFrame(note["pitch"]).to_csv(f"{prefix}pitches/{n}_{pitchAvg}_{time}.csv", index=False)

        np.savetxt(f"{prefix}harmonic/{n}_{harmonicMax}_{time}.csv", note["harmonic"], delimiter=',')
        # np.savetxt(f"{prefix}percussive/{n}_{np.max(note['percussive'])}_{time}.csv", note["percussive"], delimiter=',')