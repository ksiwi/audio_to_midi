import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from sklearn.ensemble import RandomForestClassifier

plt.figure(figsize=(10, 4))

def visualize(note):

    samplerate, data = wavfile.read(f"data/TinySOL2020/{note['Path']}")
    duration = len(data) / samplerate
    time = np.arange(0, duration, 1/samplerate)
    label = (f"{note["Instrument (in full)"]} {note["Pitch"]} {note["Dynamics"]}")
    plt.plot(time, data)
    plt.title(label)
    plt.show()
    

meta = pd.read_csv("data/TinySOL_metadata.csv")
for i in range(5):
    visualize(meta.iloc[i + 2000])

# Fast Fourier Transforms (Good for Single Instrument Recordings)

# Brass/Bass_Tuba/ordinario/BTb-ord-F#1-pp-N-N.wav