import mosqito
import os
from mosqito.utils import load
from mosqito.sq_metrics.speech_intelligibility import sii_ansi
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import pandas as pd
from scipy.signal import resample

audio_dir = '/Users/averyfield/Desktop/Bootcamp/flytechllm/~/Desktop/Bootcamp/flytechllm/inference'
audio_extensions = {'.wav', '.flac', '.ogg', '.aiff', '.mp3'}

def get_sii_input(audio, fs):
    # Resample to 48 kHz if the sampling rate is lower
    if fs < 48000:
        num_samples = int(len(audio) * 48000 / fs)
        audio = resample(audio, num_samples)
        fs = 48000
    SII, SII_spec, freq_axis = sii_ansi(audio, fs, method='critical', speech_level='normal')
    return SII

def get_sii_from_file(file):
    audio, fs = sf.read(file)
    SII, SII_spec, freq_axis = sii_ansi(audio, fs, method='critical', speech_level='normal')
    return SII

    '''
    plt.plot(freq_axis, SII_spec)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Specific value ")
    plt.title("Speech Intelligibility Index = " + f"{SII:.2f}")
    plt.show()
    '''

#Raw Audio Data
print("Input Audio Analysis:")
print('')
input_sii_data = []
from datasets import load_dataset

fleurs_asr = load_dataset("google/fleurs", "en_us")  # for English

# test out first 50 samples
audio_inputs = fleurs_asr["train"][:50]["audio"]

# Iterate through each audio input from the dataset
for audio_input in audio_inputs:
    file_path = audio_input['path']
    if os.path.isfile(file_path):
        try:
            input_sii_value = get_sii_input(file_path)
            input_sii_data.append({"filename": file_path, "SII": input_sii_value})
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    else:
        try:
            audio_array = audio_input['array']
            fs = audio_input['sampling_rate']
            input_sii_value = get_sii_input(audio_array, fs)
            input_sii_data.append({"filename": file_path, "SII": input_sii_value})
        except Exception as e:
            print(f"Error processing audio array from {file_path}: {e}")

# Create a DataFrame
input_sii_df = pd.DataFrame(input_sii_data)

# Print the DataFrame
print(input_sii_df)
plt.plot(input_sii_df['SII'])
plt.show()

#Translated Audio
sii_data = []

print('')
print("Translated Audio Analysis:")
print('')
# Iterate through each file in the directory
for audio_file in os.listdir(audio_dir):
    # Construct the full file path
    file_path = os.path.join(audio_dir, audio_file)
    # Ensure it's a file and has a valid audio extension
    if os.path.isfile(file_path) and os.path.splitext(audio_file)[1].lower() in audio_extensions:
        try:
            sii_value = get_sii_from_file(file_path)
            sii_data.append({"filename": audio_file, "SII": sii_value})
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")

# Create a DataFrame
sii_df = pd.DataFrame(sii_data)

# Print the DataFrame
print(sii_df)

