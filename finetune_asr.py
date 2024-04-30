from datasets import load_dataset, DatasetDict
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor

#load dataset

common_voice = load_dataset("mozilla-foundation/common_voice_11_0", "en", use_auth_token=False, streaming=True)
# common_voice["tuneset"] = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="validation", use_auth_token=False, streaming=True)
# common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="test", use_auth_token=False, streaming=True)

#print(common_voice)
common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])

#Load WhisperFeatureExtractor & tokenizer
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="English", task="transcribe")

'''
#testing tokenizer
input_str = common_voice["train"][0]["sentence"]
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

print(f"Input:                 {input_str}")
print(f"Decoded w/ special:    {decoded_with_special}")
print(f"Decoded w/out special: {decoded_str}")
print(f"Are equal:             {input_str == decoded_str}")
'''
#wrap feature extractor and tokenizer into processor class
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="English", task="transcribe")

#prepare data
from datasets import Audio

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

common_voice = common_voice.map(prepare_dataset)
print(common_voice["train"][0])
