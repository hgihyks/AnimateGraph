import cv2
import ffmpeg
import numpy as np
import random
import os
import subprocess

def get_video_metadata(path):
    """Gets video duration, width, and height."""
    try:
        probe = ffmpeg.probe(path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        return float(video_info['duration']), int(video_info['width']), int(video_info['height'])
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
    # Lower CRF = higher quality, larger file. 18 is high quality.
    quality_crf = 18
    # Encoding speed preset. 'medium' is a good balance.
    quality_preset = 'medium'
    # Codecs
    video_codec = 'libx264'
    audio_codec = 'aac'

    try:
        if width > target_width:
            # --- Crop ---
            print(f"Original width ({width}) > target width ({target_width}). Cropping...")
            # Read a frame to determine salient crop area
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
                # Apply quality settings to the output
                .output(output_path,
                        vcodec=video_codec,
                        acodec=audio_codec,
                        strict='experimental', # Often needed for aac
                        crf=quality_crf,      # Constant Rate Factor for quality
                        preset=quality_preset # Encoding speed/compression preset
                       )
                .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
            )
            print(f"Successfully cropped video saved to {output_path} with CRF {quality_crf}")

        elif width < target_width:
            # --- Pad ---
            print(f"Original width ({width}) < target width ({target_width}). Padding...")
            pad_total = target_width - width
            pad_left = pad_total // 2

            (
                ffmpeg
                .input(input_path)
                .filter('pad', w=target_width, h=target_height, x=pad_left, y=0, color='black')
                # Apply quality settings to the output
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
            print("Original aspect ratio is already 9:16. Re-encoding to ensure consistent quality settings...")
            # Re-encode even if dimensions match to apply desired CRF and preset
            (
                ffmpeg
                .input(input_path)
                # No filter needed, just re-encode with quality settings
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


def process_video(input_path, output_path):
    """Main processing function."""
    print(f"--- Starting processing for {input_path} ---")
    temp_clip = "temp_clip.mp4"
    # Use a distinct name for the intermediate file before final rename
    final_temp_before_rename = "final_temp_processing.mp4"

    try:
        # 1. Get metadata
        duration, width, height = get_video_metadata(input_path)
        if duration is None:
             print(f"Skipping {input_path} due to metadata error.")
             return # Exit if metadata failed

        print(f"Video Info: Duration={duration:.2f}s, Dimensions={width}x{height}")

        # 2. Determine start time
        if duration <= 10:
            start = 0
            print("Video <= 10s, using full clip.")
        else:
            start = random.uniform(0, duration - 10)
            print(f"Selected random start time: {start:.2f}s")

        # 3. Extract clip (fast, no quality loss here)
        extract_10s_clip(input_path, start, temp_clip)

        # 4. Pad/Crop/Re-encode to 9:16 with quality control
        # Pass the *original* width/height for aspect ratio calculation
        pad_to_9_16(temp_clip, final_temp_before_rename, width, height)

        # 5. Rename final file
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        os.rename(final_temp_before_rename, output_path)
        print(f"--- Successfully processed video saved to: {output_path} ---")

    except Exception as e:
        # Catch errors from called functions
        print(f"--- FAILED processing for {input_path}: {e} ---")

    finally:
        # 6. Cleanup temporary files
        if os.path.exists(temp_clip):
            try:
                os.remove(temp_clip)
                print(f"Removed temp file: {temp_clip}")
            except OSError as e:
                print(f"Error removing temp file {temp_clip}: {e}")
        # This intermediate file should only exist if an error occurred *after*
        # it was created but *before* the final rename.
        if os.path.exists(final_temp_before_rename):
            try:
                os.remove(final_temp_before_rename)
                print(f"Removed intermediate file due to error: {final_temp_before_rename}")
            except OSError as e:
                print(f"Error removing intermediate file {final_temp_before_rename}: {e}")


# --- Example Usage ---
if __name__ == "__main__":
    # Define input and output paths
    input_video = "input_videos/msn.mp4" # Make sure this file exists
    output_video = "output_videos/output_9x16_high_quality.mp4" # Changed output name

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_video)
    if output_dir and not os.path.exists(output_dir):
         os.makedirs(output_dir)
         print(f"Created output directory: {output_dir}")

    # Check if input file exists before processing
    if os.path.exists(input_video):
        process_video(input_video, output_video)
    else:
        print(f"Error: Input video file not found at '{input_video}'")
        print("Please ensure the video file exists and the path is correct.")

