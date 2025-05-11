import ffmpeg
import os
import json

def add_audio_to_video(video_path, audio_path, output_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    video_input = ffmpeg.input(video_path)
    audio_input = ffmpeg.input(audio_path)

    try:
        (
            ffmpeg
            .output(
                video_input, audio_input,
                output_path,
                vcodec='copy',
                acodec='aac',
                audio_bitrate='192k',
                shortest=None
            )
            .global_args('-map', '0:v:0', '-map', '1:a:0')  # ✅ Correct way to map
            .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        print("FFmpeg error:\n", e.stderr.decode())
        raise

# Example usage
if __name__ == "__main__":
    with open("config.json", "r") as cfg_file:
        config = json.load(cfg_file)
    video_file = config["output_video_9x16"]
    audio_file = config["narration_file"]
    output_file = config["video_file_with_audio"]
    add_audio_to_video(video_file, audio_file, output_file)
