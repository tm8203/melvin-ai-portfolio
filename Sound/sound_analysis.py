import streamlit as st
import os
import random

# Page configuration
st.set_page_config(
    page_title="Analyze Sound with Spectrogram Insights",
    layout="wide"
)

# Title and Overview
st.title("Analyze Sound with Spectrogram Insights")
st.write(
    "**Overview:** This module explores how sound files can be represented visually using spectrograms. Users can interact with the sounds, view their spectrograms, and understand how neural networks process audio data."
)
st.write(
    "**Solution:** A set of pre-recorded audio files from three categories ('dog,' 'eight,' and 'happy') are analyzed to showcase key features in sound processing and visualization. The spectrogram reveals patterns in frequency and amplitude, offering insights into sound recognition and classification."
)

# Select a category
st.header("Explore Audio Files")
category = st.selectbox("Choose a sound category:", ["dog", "eight", "happy"])

# List files in the chosen category
audio_folder = f"audio/{category}/"
audio_files = [f for f in os.listdir(audio_folder) if f.endswith(".wav")]

# Randomly select a file
selected_file = st.selectbox("Choose an audio file:", audio_files)
file_path = os.path.join(audio_folder, selected_file)

# Play the selected audio file
st.audio(file_path, format="audio/wav")

# Generate and display a spectrogram
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np

st.header("Spectrogram of the Selected Audio")
sample_rate, audio_data = wavfile.read(file_path)

# Generate spectrogram
plt.figure(figsize=(10, 4))
plt.specgram(audio_data, Fs=sample_rate, NFFT=1024, noverlap=512, cmap="viridis")
plt.title(f"Spectrogram of {selected_file}")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Intensity (dB)")
st.pyplot(plt)
