import cv2
import ffmpeg
import numpy as np
import random
import os
import json
from ultralytics import YOLO
import torchaudio

def get_video_metadata(path):
    try:
        probe = ffmpeg.probe(path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        duration_str = video_info.get('duration')
        if duration_str is None:
            duration_str = probe.get('format', {}).get('duration')
        if duration_str is None:
            raise ValueError("Could not determine video duration.")
        return float(duration_str), int(video_info['width']), int(video_info['height'])
    except Exception as e:
        return None, None, None

def find_and_save_segments(video_duration, min_duration, max_duration, total_length, output_txt_file):
    segments = []
    current_total_duration = 0.0
    num_segments = video_duration / max_duration
    print(f"num segments {num_segments}")
    print(f"total length {total_length}")
    possible_durations = [d / 2.0 for d in range(int(min_duration * 2), int(max_duration * 2) + 1)]
    for i in range(int(num_segments)):
        start_segment = i * max_duration
        end_segment = start_segment + random.choice(possible_durations)
        segments.append((start_segment, end_segment))
        current_total_duration += (end_segment - start_segment)
    
    print(f"Current total duration: {current_total_duration}")
    while current_total_duration > total_length:
        start, end = random.choice(segments)
        current_total_duration -= (end - start)
        segments.remove((start, end))
    
    segments.sort(key=lambda x: x[0])
    try:
        output_dir = os.path.dirname(output_txt_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_txt_file, 'w') as f:
            print(f"# Found Segments Total Length: {current_total_duration:.2f}s\n")
            if not segments:
                f.write("No segments found matching the criteria.\n")
            else:
                for start, end in segments:
                    f.write(f"{int(start)} - {end:.1f}\n")
    except IOError as e:
        pass

def parse_timestamps(file_path):
    segments = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.replace(',', '.').split('-')
            if len(parts) != 2:
                raise ValueError(f"Invalid line in timestamps file: {line}")
            start = float(parts[0].strip())
            end = float(parts[1].strip())
            if end <= start:
                raise ValueError(f"End time must be greater than start time: {line}")
            segments.append((start, end))
    return segments

def concatenate_segments(input_video, timestamps_file, output_video):
    segments = parse_timestamps(timestamps_file)
    video_streams = []
    for start, end in segments:
        duration = end - start
        stream = ffmpeg.input(input_video, ss=start, t=duration).video
        video_streams.append(stream)
    concatenated = ffmpeg.concat(*video_streams, v=1, a=0).node
    out = ffmpeg.output(concatenated[0], output_video)
    out.run(overwrite_output=True)

def combine_videos(input_folder, output_file):
    """
    Concatenate all video files in the input folder into a single output file.
    
    Args:
        input_folder (str): Path to the folder containing video files.
        output_file (str): Path to the output video file.
    
    Returns:
        str: Path to the output file.
    
    Raises:
        ValueError: If no video files are found in the input folder.
        ffmpeg.Error: If FFmpeg processing fails.
    """
    # Temporary file to list video inputs
    temp_list = 'inputs.txt'
    
    # Get sorted list of video files
    files = sorted(os.listdir(input_folder))
    video_files = [f for f in files if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]
    
    # Check if there are any video files
    if not video_files:
        raise ValueError("No video files found in the input folder.")
    
    # Write video file paths to temporary list file
    with open(temp_list, 'w') as f:
        for vid in video_files:
            full_path = os.path.abspath(os.path.join(input_folder, vid))
            f.write(f"file '{full_path}'\n")
    
    try:
        # Set up FFmpeg input using concat demuxer
        stream = ffmpeg.input(temp_list, format='concat', safe=0)
        
        # Configure output with re-encoding
        stream = stream.output(
            output_file,
            vcodec='libx264',    # Video codec: H.264
            r=24,             # Frame rate: 30 fps
            video_bitrate='5000k',      # Video bitrate: 5000k
            preset='medium'   # Encoding preset: Balance speed and quality
        )
        print(stream.get_args())
        
        # Run FFmpeg command, overwriting output if it exists
        stream.run(overwrite_output=True)
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_list):
            os.remove(temp_list)
    
    return output_file

def smart_crop_box(frame, model, target_aspect_ratio=9/16):
    h, w, _ = frame.shape
    results = model(frame, verbose=False)[0]

    if results.boxes:
        boxes = results.boxes.xyxy.cpu().numpy()
        x_centers = [(box[0] + box[2]) / 2 for box in boxes]
        y_centers = [(box[1] + box[3]) / 2 for box in boxes]
        center_x = int(sum(x_centers) / len(x_centers))
        center_y = int(sum(y_centers) / len(y_centers))
    else:
        center_x = w // 2
        center_y = h // 2

    if w / h > target_aspect_ratio:
        crop_h = h
        crop_w = int(h * target_aspect_ratio)
    else:
        crop_w = w
        crop_h = int(w / target_aspect_ratio)

    x1 = max(0, min(w - crop_w, center_x - crop_w // 2))
    y1 = max(0, min(h - crop_h, center_y - crop_h // 2))
    
    crop_w = crop_w - (crop_w % 2)
    crop_h = crop_h - (crop_h % 2)
    x1 = max(0, min(w - crop_w, center_x - crop_w // 2))
    y1 = max(0, min(h - crop_h, center_y - crop_h // 2))
    return int(x1), int(y1), int(crop_w), int(crop_h)

def crop_video(input_path, output_path, model_path="yolov8n.pt"):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    temp_frames_dir = "temp_frames"
    os.makedirs(temp_frames_dir, exist_ok=True)

    frame_paths = []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if i == 0:
            x, y, cw, ch = smart_crop_box(frame, model)

        cropped = frame[y:y+ch, x:x+cw]
        frame_path = os.path.join(temp_frames_dir, f"frame_{i:05d}.png")
        cv2.imwrite(frame_path, cropped)
        frame_paths.append(frame_path)
        i += 1

    cap.release()

    (
        ffmpeg
        .input(os.path.join(temp_frames_dir, 'frame_%05d.png'), framerate=fps)
        .output(output_path, vcodec='libx264', pix_fmt='yuv420p')
        .run(overwrite_output=True)
    )

    for f in frame_paths:
        os.remove(f)
    os.rmdir(temp_frames_dir)

def get_audio_duration(file_path: str) -> float:
    """
    Returns the duration of an audio file in seconds.

    Parameters:
    - file_path (str): Path to the audio file.

    Returns:
    - float: Duration in seconds.
    """
    waveform, sample_rate = torchaudio.load(file_path)
    duration = waveform.size(1) / sample_rate
    return int(duration) + 2

if __name__ == "__main__":
    with open('config.json', 'r') as cfg_file:
        config = json.load(cfg_file)
    input_folder = config['input_folder']
    output_file = config['output_file']
    output_video_9x16 = config['output_video_9x16']
    output_segments_file = config['output_segments_file']
    segment_min_duration = config['segment_min_duration']
    segment_max_duration = config['segment_max_duration']
    segment_total_length = get_audio_duration(config["narration_file"]) + segment_max_duration
    segmented_video = config['segmented_video_path']
    output_dir = os.path.dirname(output_video_9x16)
    # combined_video_path = combine_videos(input_folder, output_file)
    
    original_duration, _, _ = get_video_metadata(output_file)
    find_and_save_segments(
            original_duration,
            segment_min_duration,
            segment_max_duration,
            segment_total_length,
            output_segments_file)
    
    # concatenate_segments(output_file, output_segments_file, segmented_video)
    # crop_video(segmented_video, output_video_9x16)
    