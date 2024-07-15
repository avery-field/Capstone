from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelWithLMHead,
    TranslationPipeline,
    FastSpeech2ConformerHifiGan,
)


# ASR English
audio2text = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny.en",
    chunk_length_s=30,
) 


translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es", device='mps')

# Text2Audio ES to Audio
text2audio = pipeline(model="suno/bark-small")

#writing translations as csv to evaluate translation quality
import soundfile as sf
import os
import csv

def add_text_to_csv(text, csv_file_path):
    # Open the CSV file for appending
    with open(csv_file_path, mode='a', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        # Write the text to the CSV
        writer.writerow([text])

output_csv_file = 'translations.csv'

def translate(input_path, output_path):
    english_text = audio2text(input_path, batch_size=8)
    spanish_text = translator(english_text['text'])
    spanish_text = spanish_text[0]['translation_text']
    add_text_to_csv(spanish_text, output_csv_file)

    speech = text2audio(spanish_text, generate_kwargs={"temperature": 0.7})

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    sf.write(
       output_path, speech["audio"].squeeze(), samplerate=speech["sampling_rate"]
    )



from datasets import load_dataset

fleurs_asr = load_dataset("google/fleurs", "en_us")  # for English

# test out first 50 samples
audio_inputs = fleurs_asr["train"][:50]["audio"]

output_directory = '~/Desktop/Bootcamp/flytechllm/inference'

for i, audio_input in enumerate(audio_inputs):
    output_path = os.path.join(output_directory, f'translated_audio_{i}.wav')
    translate(audio_input, output_path)