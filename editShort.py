import cv2
import ffmpeg
import numpy as np
import random
import os
import subprocess
import math
import json
from ultralytics import YOLO

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

def extract_10s_clip(input_path, start_time, temp_path):
    try:
        (
            ffmpeg
            .input(input_path, ss=start_time, t=10)
            .output(temp_path, c='copy')
            .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise

def get_salient_crop_x(frame, target_width):
    try:
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        success, saliencyMap = saliency.computeSaliency(frame)
        if not success or saliencyMap is None:
            center_x = frame.shape[1] // 2
        else:
            saliencyMap = (saliencyMap * 255).astype(np.uint8)
            moments = cv2.moments(saliencyMap)
            if moments["m00"] == 0:
                center_x = frame.shape[1] // 2
            else:
                center_x = int(moments["m10"] / moments["m00"])
        x1 = max(0, center_x - target_width // 2)
        x2 = min(frame.shape[1], x1 + target_width)
        if x2 - x1 < target_width:
            x1 = max(0, x2 - target_width)
        return x1, x2
    except Exception as e:
        center_x = frame.shape[1] // 2
        x1 = max(0, center_x - target_width // 2)
        x2 = min(frame.shape[1], x1 + target_width)
        if x2 - x1 < target_width:
            x1 = max(0, x2 - target_width)
        return x1, x2

def pad_to_9_16(input_path, output_path, width, height):
    target_height = height
    target_width = int(target_height * 9 / 16)
    quality_crf = 18
    quality_preset = 'medium'
    video_codec = 'libx264'
    audio_codec = 'aac'
    try:
        clip_probe = ffmpeg.probe(input_path)
        clip_video_info = next(s for s in clip_probe['streams'] if s['codec_type'] == 'video')
        clip_width = int(clip_video_info['width'])
        clip_height = int(clip_video_info['height'])
        target_height = clip_height
        target_width = int(target_height * 9 / 16)
        if clip_width > target_width:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise IOError(f"Cannot open video file: {input_path}")
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                raise IOError(f"Cannot read frame from: {input_path}")
            x1, x2 = get_salient_crop_x(frame, target_width)
            (
                ffmpeg
                .input(input_path)
                .filter('crop', w=target_width, h=target_height, x=x1, y=0)
                .output(output_path,
                        vcodec=video_codec,
                        acodec=audio_codec,
                        strict='experimental',
                        crf=quality_crf,
                        preset=quality_preset
                       )
                .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
            )
        elif clip_width < target_width:
            pad_total = target_width - clip_width
            pad_left = pad_total // 2
            (
                ffmpeg
                .input(input_path)
                .filter('pad', w=target_width, h=target_height, x=pad_left, y=0, color='black')
                .output(output_path,
                        vcodec=video_codec,
                        acodec=audio_codec,
                        strict='experimental',
                        crf=quality_crf,
                        preset=quality_preset
                       )
                .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
            )
        else:
            (
                ffmpeg
                .input(input_path)
                .output(output_path,
                        vcodec=video_codec,
                        acodec=audio_codec,
                        strict='experimental',
                        crf=quality_crf,
                        preset=quality_preset
                       )
                .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
            )
    except ffmpeg.Error as e:
        raise
    except Exception as e:
        raise

def find_and_save_segments(video_duration, min_duration, max_duration, total_length, output_txt_file):
    segments = []
    current_total_duration = 0.0
    num_segments = video_duration / max_duration
    possible_durations = [d / 2.0 for d in range(int(min_duration * 2), int(max_duration * 2) + 1)]
    for i in range(int(num_segments)):
        start_segment = i * max_duration
        end_segment = start_segment + random.choice(possible_durations)
        segments.append((start_segment, end_segment))
        current_total_duration += (end_segment - start_segment)
    while current_total_duration > total_length:
        start, end = random.choice(segments)
        current_total_duration -= (end - start)
        segments.remove((start, end))
    if not segments:
        pass
    elif abs(current_total_duration - total_length) > 0.25 and current_total_duration < total_length:
        pass
    segments.sort(key=lambda x: x[0])
    try:
        output_dir = os.path.dirname(output_txt_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_txt_file, 'w') as f:
            print(f"# Found Segments Total Length: {current_total_duration:.2f}s\n")
            # f.write(f"# Segments for video (Total Duration: {video_duration:.2f}s)\n")
            # f.write(f"# Target Total Length: {total_length:.2f}s (Capped at: {total_length:.2f}s)\n")
            # f.write(f"# Found Segments Total Length: {current_total_duration:.2f}s\n")
            # f.write("# Format: Start Time (s) - End Time (s)\n")
            # f.write("# Constraints: Integer Start, 0.5s Multiple Duration\n")
            # f.write("----------------------------------------\n")
            if not segments:
                f.write("No segments found matching the criteria.\n")
            else:
                for start, end in segments:
                    f.write(f"{int(start)} - {end:.1f}\n")
    except IOError as e:
        pass

def process_video(input_path, output_path):
    temp_clip = "temp_clip_for_9x16.mp4"
    final_temp_before_rename = "final_temp_9x16_processing.mp4"
    try:
        duration, width, height = get_video_metadata(input_path)
        if duration is None:
            return
        if duration <= 10:
            start = 0
        else:
            start = random.uniform(0, duration - 10)
        extract_10s_clip(input_path, start, temp_clip)
        pad_to_9_16(temp_clip, final_temp_before_rename, width, height)
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        os.rename(final_temp_before_rename, output_path)
    except Exception as e:
        pass
    finally:
        if os.path.exists(temp_clip):
            try:
                os.remove(temp_clip)
            except OSError as e:
                pass
        if os.path.exists(final_temp_before_rename):
            try:
                os.remove(final_temp_before_rename)
            except OSError as e:
                pass

def combine_videos(input_folder, output_file):
    temp_list = 'inputs.txt'
    files = sorted(os.listdir(input_folder))
    video_files = [f for f in files if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]
    with open(temp_list, 'w') as f:
        for vid in video_files:
            f.write(f"file '{os.path.abspath(os.path.join(input_folder, vid))}'\n")
    ffmpeg.input(temp_list, format='concat', safe=0).output(output_file, c='copy').run()
    os.remove(temp_list)
    return output_file

def parse_timestamps(file_path):
    """
    Parse a text file with lines like 'start - end' into a list of (start, end) tuples.
    """
    segments = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Expect format: start - end
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
    # Parse segments
    segments = parse_timestamps(timestamps_file)

    # Create a list of trimmed video streams only
    video_streams = []
    for start, end in segments:
        duration = end - start
        stream = ffmpeg.input(input_video, ss=start, t=duration).video
        video_streams.append(stream)

    # Concatenate video streams only
    concatenated = ffmpeg.concat(*video_streams, v=1, a=0).node

    # Output the result (video only)
    out = ffmpeg.output(concatenated[0], output_video)
    out.run(overwrite_output=True)

def smart_crop_box(frame, model, target_aspect_ratio=9/16):
    h, w, _ = frame.shape
    results = model(frame, verbose=False)[0]

    # If detections exist, calculate bounding box center of all objects
    if results.boxes:
        boxes = results.boxes.xyxy.cpu().numpy()
        x_centers = [(box[0] + box[2]) / 2 for box in boxes]
        y_centers = [(box[1] + box[3]) / 2 for box in boxes]
        center_x = int(sum(x_centers) / len(x_centers))
        center_y = int(sum(y_centers) / len(y_centers))
    else:
        # No detections, fallback to center
        center_x = w // 2
        center_y = h // 2

    # Compute target crop dimensions
    if w / h > target_aspect_ratio:
        # Wider than 9:16 → crop width
        crop_h = h
        crop_w = int(h * target_aspect_ratio)
    else:
        # Taller than 9:16 → crop height
        crop_w = w
        crop_h = int(w / target_aspect_ratio)

    # Ensure crop box is within frame
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

    # Use ffmpeg to combine images into video
    (
        ffmpeg
        .input(os.path.join(temp_frames_dir, 'frame_%05d.png'), framerate=fps)
        .output(output_path, vcodec='libx264', pix_fmt='yuv420p')
        .run(overwrite_output=True)
    )

    # Cleanup
    for f in frame_paths:
        os.remove(f)
    os.rmdir(temp_frames_dir)

if __name__ == "__main__":
    with open('config.json', 'r') as cfg_file:
        config = json.load(cfg_file)
    input_folder = config['input_folder']
    output_file = config['output_file']
    output_video_9x16 = config['output_video_9x16']
    output_segments_file = config['output_segments_file']
    segment_min_duration = config['segment_min_duration']
    segment_max_duration = config['segment_max_duration']
    segment_total_length = config['segment_total_length']
    segmented_video = config['segmented_video_path']
    output_dir = os.path.dirname(output_video_9x16)
    combined_video_path = combine_videos(input_folder, output_file)
    
    original_duration, _, _ = get_video_metadata(combined_video_path)
    find_and_save_segments(
            original_duration,
            segment_min_duration,
            segment_max_duration,
            segment_total_length,
            output_segments_file)
    
    concatenate_segments(output_file, output_segments_file, segmented_video)
    
    # Crop segmented video to 9:16 aspect ratio and save as final output
    # seg_duration, seg_width, seg_height = get_video_metadata(segmented_video)
    # pad_to_9_16(segmented_video, output_video_9x16, seg_width, seg_height)
    crop_video(segmented_video, output_video_9x16)
    