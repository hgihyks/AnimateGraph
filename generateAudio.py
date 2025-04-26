# Import Bark
from bark import SAMPLE_RATE, generate_audio
import scipy

# Text you want to synthesize
# --- Load text prompt from file ---
with open("input/prompt.txt", "r", encoding="utf-8") as f:
    text_prompt = f.read().strip()

speaker = "custom_voices/en_female_serious_reader.npz"

# Generate audio array
audio_array = generate_audio(text_prompt, history_prompt=speaker)

# Save to WAV file
output_path = "output/bark_output.wav"
scipy.io.wavfile.write(output_path, SAMPLE_RATE, audio_array)

print("Audio generated and saved as 'bark_output.wav'.")
