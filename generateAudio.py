from bark import SAMPLE_RATE, generate_audio
import scipy
import numpy as np
import re
import json

# Load config
with open("config.json", "r") as cfg_file:
    config = json.load(cfg_file)

# Load text prompt from file
prompt_file = config.get("prompt_file")
with open(prompt_file, "r", encoding="utf-8") as f:
    text_prompt = f.read().strip()

# Split into sentences using regex, then group them into chunks of approximately 30 words
sentences = re.split(r'(?<=[.!?]) +', text_prompt)
chunks = []
current_chunk = ""
current_word_count = 0

for sentence in sentences:
    word_count = len(sentence.split())
    if current_word_count + word_count <= 30:
        current_chunk += (" " if current_chunk else "") + sentence
        current_word_count += word_count
    else:
        if current_chunk:
            chunks.append(current_chunk.strip())
        current_chunk = sentence
        current_word_count = word_count

# Add the last chunk
if current_chunk:
    chunks.append(current_chunk.strip())

# print(chunks)

# Use custom community-trained voice
speaker = config["speaker"]

# Generate audio for each chunk and concatenate
all_audio = []
for chunk in chunks:
    audio_chunk = generate_audio(chunk, history_prompt=speaker)
    all_audio.append(audio_chunk)

# Concatenate all audio chunks
final_audio = np.concatenate(all_audio)

# Save the generated audio to a WAV file
output_path = config["audio_output_path"]
scipy.io.wavfile.write(output_path, SAMPLE_RATE, final_audio)

print(f"Audio generated and saved as '{output_path}'.")
