import whisper_timestamped as whisper
import os
import subprocess
from pathlib import Path
import json

# === LOAD CONFIG ===
with open("caption_config.json") as config_file:
    config = json.load(config_file)

VIDEO_PATH = config["video_path"]
OUTPUT_PATH = config["output_path"]
FONT_NAME = config["font_name"]
FONT_SIZE = config["font_size"]
OUTLINE = config["outline"]
TEXT_COLOR = config["text_color"]
OUTLINE_COLOR = config["outline_color"]
ALIGNMENT = config["alignment"]  # 1=bottom-left, 2=bottom-center, 3=bottom-right
LINE_DURATION = config["min_line_duration"]
BG_COLOR = config["bg_color"]  # Not fully supported by ASS but placeholder
WORDS_PER_CAPTION = config.get("words_per_caption")
POS_X = config.get("pos_x")
POS_Y = config.get("pos_y")

# === TRANSCRIBE AUDIO ===
print("Transcribing with whisper-timestamped...")
model = whisper.load_model("base")
result = whisper.transcribe(model, VIDEO_PATH)
segments = result["segments"]

# === CONVERT TO .ASS FORMAT ===
ass_path = Path(VIDEO_PATH).with_suffix(".ass")
with open(ass_path, "w") as f:
    f.write(f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{FONT_NAME},{FONT_SIZE},{TEXT_COLOR},{OUTLINE_COLOR},0,0,1,{OUTLINE},0,{ALIGNMENT},0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""")

    def format_time(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        cs = int((seconds % 1) * 100)
        return f"{h:01}:{m:02}:{s:02}.{cs:02}"

    for seg in segments:
        words = seg["words"]
        group = []

        for word in words:
            if word.get("text", '').strip():
                group.append(word)

                if len(group) == WORDS_PER_CAPTION:
                    start = format_time(group[0]["start"])
                    end = format_time(group[-1]["end"])
                    text = " ".join(w["text"].strip().replace("{", "").replace("}", "") for w in group)

                    pos = fr"{{\pos({POS_X},{POS_Y})}}"
                    f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{pos}{text}\n")

                    group = []  # reset group

        # Handle leftover words (if any words are left ungrouped)
        if group:
            start = format_time(group[0]["start"])
            end = format_time(group[-1]["end"])
            text = " ".join(w["text"].strip().replace("{", "").replace("}", "") for w in group)

            pos = fr"{{\pos({POS_X},{POS_Y})}}"
            f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{pos}{text}\n")

# === BURN SUBTITLES ===
output_path = OUTPUT_PATH
print("Burning subtitles with FFmpeg...")
subprocess.run([
    "ffmpeg", "-i", VIDEO_PATH,
    "-vf", f"subtitles={ass_path}",
    "-c:a", "copy", output_path
])

print(f"Done! Output saved as: {output_path}")
