# %%
import torch
import whisper

# %%
model_names = [
    'tiny.en',
    'tiny',
    'base.en',
    'base',
    'small.en',
    'small',
    'medium.en',
    'medium',
    'large']
model_name = "base"

audio_file = "/home/boren/data/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac"
model = whisper.load_model(model_name, download_root="data/models")
# model_file = f"/home/boren/whisper/data/models/{model_name}.pt"
# state_dict = torch.load(model_file, map_location="cpu")
# %%
result = model.transcribe(audio_file, language="en", beam_size=4)
# result = model.transcribe(audio_file)
print(result["text"])
