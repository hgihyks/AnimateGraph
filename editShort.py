import cv2
import ffmpeg
import numpy as np
import random
import os
import subprocess
import math # Needed for ceiling and floor functions

def get_video_metadata(path):
    """Gets video duration, width, and height."""
    try:
        probe = ffmpeg.probe(path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        # Ensure duration is present and valid
        duration_str = video_info.get('duration')
        if duration_str is None:
            print(f"Warning: Duration not found in metadata for {path}. Trying 'format' duration.")
            duration_str = probe.get('format', {}).get('duration')

        if duration_str is None:
             raise ValueError("Could not determine video duration.")

        return float(duration_str), int(video_info['width']), int(video_info['height'])
    except Exception as e:
        print(f"Error getting metadata for {path}: {e}")
        # Indicate failure
        return None, None, None

def extract_10s_clip(input_path, start_time, temp_path):
    """Extracts a 10s clip using stream copy (fast, preserves quality)."""
    try:
        (
            ffmpeg
            .input(input_path, ss=start_time, t=10)
            .output(temp_path, c='copy') # Copy codecs, no re-encoding here
            .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        )
        print(f"Successfully extracted 10s clip to {temp_path}")
    except ffmpeg.Error as e:
        print(f"FFmpeg error during clip extraction: {e.stderr.decode()}")
        raise # Propagate error

def get_salient_crop_x(frame, target_width):
    """Finds the horizontal crop region based on saliency."""
    try:
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        success, saliencyMap = saliency.computeSaliency(frame)
        if not success or saliencyMap is None:
            print("Saliency detection failed, using center crop.")
            center_x = frame.shape[1] // 2
        else:
            # Use moments to find the center of the saliency map
            saliencyMap = (saliencyMap * 255).astype(np.uint8)
            moments = cv2.moments(saliencyMap)
            if moments["m00"] == 0:
                 print("Saliency map empty, using center crop.")
                 center_x = frame.shape[1] // 2
            else:
                center_x = int(moments["m10"] / moments["m00"])
            print(f"Saliency center detected at x={center_x}")

        # Calculate crop coordinates
        x1 = max(0, center_x - target_width // 2)
        x2 = min(frame.shape[1], x1 + target_width)
        # Ensure the width is exactly target_width
        if x2 - x1 < target_width:
            x1 = max(0, x2 - target_width)
        print(f"Cropping from x={x1} to x={x2}")
        return x1, x2
    except Exception as e:
        print(f"Error in saliency detection: {e}. Using center crop.")
        # Fallback to center crop on any error
        center_x = frame.shape[1] // 2
        x1 = max(0, center_x - target_width // 2)
        x2 = min(frame.shape[1], x1 + target_width)
        if x2 - x1 < target_width:
            x1 = max(0, x2 - target_width)
        return x1, x2


def pad_to_9_16(input_path, output_path, width, height):
    """
    Pads or crops the video to 9:16 aspect ratio, controlling output quality.
    """
    target_height = height # Keep original height for aspect ratio calculation
    target_width = int(target_height * 9 / 16)
    print(f"Target 9:16 dimensions based on original height: {target_width}x{target_height}")

    # --- Define desired output quality settings ---
    quality_crf = 18
    quality_preset = 'medium'
    video_codec = 'libx264'
    audio_codec = 'aac'

    try:
        # Determine original width of the *clip* (input_path is the temp clip)
        clip_probe = ffmpeg.probe(input_path)
        clip_video_info = next(s for s in clip_probe['streams'] if s['codec_type'] == 'video')
        clip_width = int(clip_video_info['width'])
        clip_height = int(clip_video_info['height'])

        # Recalculate target_width based on clip's actual height
        target_height = clip_height
        target_width = int(target_height * 9 / 16)

        if clip_width > target_width:
            # --- Crop ---
            print(f"Clip width ({clip_width}) > target width ({target_width}). Cropping...")
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
            print(f"Successfully cropped video saved to {output_path} with CRF {quality_crf}")

        elif clip_width < target_width:
            # --- Pad ---
            print(f"Clip width ({clip_width}) < target width ({target_width}). Padding...")
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
            print(f"Successfully padded video saved to {output_path} with CRF {quality_crf}")

        else:
            # --- Already 9:16 ---
            print("Clip aspect ratio is already 9:16. Re-encoding for consistent quality...")
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
            print(f"Successfully re-encoded video saved to {output_path} with CRF {quality_crf}")

    except ffmpeg.Error as e:
        print(f"FFmpeg error during padding/cropping/re-encoding: {e.stderr.decode()}")
        raise # Propagate error
    except Exception as e:
        print(f"Error during padding/cropping/re-encoding: {e}")
        raise # Propagate error

def round_to_nearest_half(n):
    """Rounds a number to the nearest multiple of 0.5."""
    # Ensure positive results for small negative inputs if needed, though duration should be positive
    return round(n * 2) / 2

def find_and_save_segments(video_duration, min_duration, max_duration, total_length, output_txt_file):
    """
    Finds non-overlapping segments with integer start times and 0.5s multiple durations,
    aiming for even distribution and precise total length, and saves them to a text file.
    (Revised logic: pick start time first, then find valid duration)

    Args:
        video_duration (float): The total duration of the source video in seconds.
        min_duration (float): The minimum duration of a single segment in seconds.
        max_duration (float): The maximum duration of a single segment in seconds.
        total_length (float): The target total duration of all segments combined.
        output_txt_file (str): The path to the text file where segments will be saved.
    """
    print(f"\n--- Finding Segments (Revised Logic: Integer Start, 0.5s Duration, Precise Total Length) ---")
    print(f"Parameters: Video Duration={video_duration:.2f}s, Min Segment={min_duration:.2f}s, Max Segment={max_duration:.2f}s, Target Total={total_length:.2f}s")

    # --- Input Validation ---
    if min_duration <= 0 or max_duration <= 0 or total_length <= 0:
        print("Error: Durations and total length must be positive.")
        return
    if min_duration > max_duration:
        print("Error: Minimum duration cannot be greater than maximum duration.")
        return
    # Use rounded durations for calculations involving constraints
    min_duration_r = max(0.5, round_to_nearest_half(min_duration))
    max_duration_r = round_to_nearest_half(max_duration)
    if min_duration_r > max_duration_r:
         print(f"Error: Rounded min_duration ({min_duration_r}) > rounded max_duration ({max_duration_r}).")
         return
    print(f"Adjusted Constraints: Min Duration={min_duration_r:.1f}s, Max Duration={max_duration_r:.1f}s")

    # Store original target before potentially capping it
    original_target_total_length = total_length
    if total_length > video_duration:
        print(f"Warning: Target total length ({total_length:.2f}s) is greater than video duration ({video_duration:.2f}s). Capping at video duration.")
        total_length = video_duration
    # Cap max_duration_r if it exceeds video duration
    max_duration_r = min(max_duration_r, video_duration)
    if min_duration_r > max_duration_r:
         print("Error: Adjusted max_duration_r is now less than min_duration_r after capping.")
         return

    # --- Initialization ---
    segments = []
    current_total_duration = 0.0
    num_segments = video_duration / max_duration_r
    possible_durations = [d / 2.0 for d in range(int(min_duration_r * 2), int(max_duration_r * 2) + 1)]



    # --- Segment Placement Loop ---
    
    for i in range(int(num_segments)) :
        start_segment = i * max_duration_r
        end_segment = start_segment + random.choice(possible_durations)
        segments.append((start_segment, end_segment))
        current_total_duration += (end_segment - start_segment)

    while current_total_duration > total_length :
        start, end = random.choice(segments)
        current_total_duration -= (end - start)
        segments.remove((start, end))


    # --- Finalization ---
    if not segments:
         print("No segments were placed.")
    # Check if final duration is significantly off target
    elif abs(current_total_duration - total_length) > 0.25 and current_total_duration < total_length : # Allow deviation up to half a segment step
         print(f"Warning: Could only place {current_total_duration:.2f}s of segments, target was {total_length:.2f}s. Constraints might be too strict or got stuck.")

    # Sort segments by start time
    segments.sort(key=lambda x: x[0])

    # Write segments to the output file
    try:
        output_dir = os.path.dirname(output_txt_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")

        with open(output_txt_file, 'w') as f:
            f.write(f"# Segments for video (Total Duration: {video_duration:.2f}s)\n")
            f.write(f"# Target Total Length: {original_target_total_length:.2f}s (Capped at: {total_length:.2f}s)\n")
            f.write(f"# Found Segments Total Length: {current_total_duration:.2f}s\n")
            f.write("# Format: Start Time (s) - End Time (s)\n")
            f.write("# Constraints: Integer Start, 0.5s Multiple Duration\n")
            f.write("----------------------------------------\n")
            if not segments:
                f.write("No segments found matching the criteria.\n")
            else:
                for start, end in segments:
                    # Format start as integer, end with one decimal place
                    f.write(f"{int(start)} - {end:.1f}\n")
        print(f"Successfully wrote {len(segments)} segments to {output_txt_file}")
        print(f"Achieved total segment duration: {current_total_duration:.1f}s") # Show .1f precision

    except IOError as e:
        print(f"Error writing segments to file {output_txt_file}: {e}")

    print("--- Finished Finding Segments ---")


def process_video(input_path, output_path):
    """Main processing function for 10s clip extraction and 9:16 conversion."""
    print(f"--- Starting 9:16 Clip Processing for {input_path} ---")
    temp_clip = "temp_clip_for_9x16.mp4" # Use specific temp name
    final_temp_before_rename = "final_temp_9x16_processing.mp4"

    try:
        # 1. Get metadata (needed for original dimensions)
        duration, width, height = get_video_metadata(input_path)
        if duration is None:
             print(f"Skipping 9:16 processing for {input_path} due to metadata error.")
             return # Exit if metadata failed

        print(f"Video Info: Duration={duration:.2f}s, Dimensions={width}x{height}")

        # 2. Determine start time for the 10s clip
        if duration <= 10:
            start = 0
            print("Video <= 10s, using full clip for 9:16 processing.")
        else:
            # Use a random start for the 10s clip
            start = random.uniform(0, duration - 10)
            print(f"Selected random start time for 10s clip: {start:.2f}s")

        # 3. Extract clip (fast, no quality loss here)
        extract_10s_clip(input_path, start, temp_clip)

        # 4. Pad/Crop/Re-encode the 10s clip to 9:16 with quality control
        # Pass the *original* width/height for aspect ratio calculation base
        pad_to_9_16(temp_clip, final_temp_before_rename, width, height)

        # 5. Rename final file
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        os.rename(final_temp_before_rename, output_path)
        print(f"--- Successfully processed 9:16 clip saved to: {output_path} ---")

    except Exception as e:
        # Catch errors from called functions
        print(f"--- FAILED 9:16 Clip Processing for {input_path}: {e} ---")
        # Optionally re-raise the error if needed: raise e

    finally:
        # 6. Cleanup temporary files
        if os.path.exists(temp_clip):
            try:
                os.remove(temp_clip)
                print(f"Removed temp file: {temp_clip}")
            except OSError as e:
                print(f"Error removing temp file {temp_clip}: {e}")
        if os.path.exists(final_temp_before_rename):
            try:
                os.remove(final_temp_before_rename)
                print(f"Removed intermediate file due to error or completion: {final_temp_before_rename}")
            except OSError as e:
                print(f"Error removing intermediate file {final_temp_before_rename}: {e}")


# --- Example Usage ---
if __name__ == "__main__":
    # --- Configuration ---
    input_video = "input_videos/msn.mp4" # Make sure this file exists
    output_video_9x16 = "output_videos/output_9x16_high_quality.mp4"
    output_segments_file = "output_videos/segments_integer_start.txt" # Reverted output name

    # Segment finding parameters
    segment_min_duration = 1.0  # seconds
    segment_max_duration = 3.0  # seconds
    segment_total_length = 30.0 # seconds

    # --- Preparations ---
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_video_9x16) # Use one of the output paths
    if output_dir and not os.path.exists(output_dir):
         os.makedirs(output_dir)
         print(f"Created output directory: {output_dir}")

    # --- Execution ---
    # Check if input file exists before processing
    if os.path.exists(input_video):
        # 1. Find and save segments (using original video duration)
        print("\n>>> Step 1: Finding Segments <<<")
        original_duration, _, _ = get_video_metadata(input_video)
        if original_duration is not None:
            find_and_save_segments(
                video_duration=original_duration,
                min_duration=segment_min_duration,
                max_duration=segment_max_duration,
                total_length=segment_total_length,
                output_txt_file=output_segments_file
            )
        else:
            print(f"Skipping segment finding for {input_video} due to metadata error.")

        # 2. Process video for 9:16 clip (optional, based on original script)
        # print("\n>>> Step 2: Processing 9:16 Clip <<<")
        # process_video(input_video, output_video_9x16) # Commented out for faster segment testing

    else:
        print(f"Error: Input video file not found at '{input_video}'")
        print("Please ensure the video file exists and the path is correct.")

