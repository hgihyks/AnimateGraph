import ffmpeg
import os
import shutil
import json

def process_video(video_path, timestamps_path, output_path):
    """Process video by stacking clips vertically based on timestamps using FFmpeg."""
    # Ensure input files exist
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found.")
        return
    if not os.path.exists(timestamps_path):
        print(f"Error: Timestamps file {timestamps_path} not found.")
        return

    # Get video duration for validation
    try:
        probe = ffmpeg.probe(video_path)
        duration = float(probe['format']['duration'])
    except ffmpeg.Error as e:
        print(f"Error probing video: {e.stderr.decode()}")
        return

    # Read timestamps from file
    with open(timestamps_path, 'r') as f:
        lines = f.readlines()
    
    # Prepare temporary directory for segments
    temp_dir = "temp_segments"
    os.makedirs(temp_dir, exist_ok=True)
    
    segment_files = []
    for i, line in enumerate(lines):
        # Parse timestamp range
        try:
            n_str, m_str = line.strip().split('-')
            n = float(n_str.strip())
            m = float(m_str.strip())
        except ValueError:
            print(f"Skipping invalid timestamp line {i+1}: {line.strip()}")
            continue
        
        # Validate timestamps
        m_adjusted = min(m, duration)
        bottom_start = n + 0.5
        bottom_end = min(m_adjusted + 0.5, duration)
        
        if m_adjusted <= n or bottom_start >= duration or bottom_end <= bottom_start:
            print(f"Skipping segment {i+1}: invalid range {n}-{m} (adjusted: {n}-{m_adjusted}, bottom: {bottom_start}-{bottom_end})")
            continue

        # Define segment path
        segment_path = os.path.join(temp_dir, f"segment_{i}.mp4")
        
        # Construct filter complex for trimming, cropping, scaling, and stacking
        filter_complex = (
            f"[0:v]trim=start={n}:end={m_adjusted},setpts=PTS-STARTPTS,crop='min(iw,ih*9/8)':'min(ih,iw*8/9)':'(iw-ow)/2':'(ih-oh)/2',scale=1080:960[top];"
            f"[0:v]trim=start={bottom_start}:end={bottom_end},setpts=PTS-STARTPTS,crop='min(iw,ih*9/8)':'min(ih,iw*8/9)':'(iw-ow)/2':'(ih-oh)/2',scale=1080:960[bottom];"
            f"[top][bottom]vstack=inputs=2[v]"
        )
        
        # Run FFmpeg to create the segment
        try:
            (
                ffmpeg
                .input(video_path)
                .output(segment_path, filter_complex=filter_complex, map='[v]', an=None, vcodec='libx264', pix_fmt='yuv420p')
                .run(overwrite_output=True, capture_stderr=True)
            )
            
            # Verify segment validity
            try:
                segment_probe = ffmpeg.probe(segment_path)
                video_info = next((stream for stream in segment_probe['streams'] if stream['codec_type'] == 'video'), None)
                if not video_info:
                    print(f"Segment {i+1} discarded: no video stream found")
                    if os.path.exists(segment_path):
                        os.remove(segment_path)
                    continue
                
                segment_duration = float(segment_probe['format']['duration'])
                width = int(video_info['width'])
                height = int(video_info['height'])
                
                if segment_duration <= 0 or width != 1080 or height != 1920:
                    print(f"Segment {i+1} discarded: invalid duration ({segment_duration}s) or resolution ({width}x{height})")
                    if os.path.exists(segment_path):
                        os.remove(segment_path)
                    continue
                
                print(f"Segment {i+1} created: duration={segment_duration}s, resolution={width}x{height}")
                segment_files.append(segment_path)
                
            except (ffmpeg.Error, StopIteration) as e:
                print(f"Segment {i+1} discarded: failed to probe ({str(e)})")
                if os.path.exists(segment_path):
                    os.remove(segment_path)
                
        except ffmpeg.Error as e:
            print(f"Error processing segment {i+1}: {e.stderr.decode()}")
            if os.path.exists(segment_path):
                os.remove(segment_path)
    
    if segment_files:
        # Create a file list for concatenation
        concat_list_path = "concat_list.txt"
        with open(concat_list_path, "w") as f:
            for segment in segment_files:
                f.write(f"file '{segment}'\n")
        
        # Concatenate all segments
        try:
            (
                ffmpeg
                .input(concat_list_path, format='concat', safe=0)
                .output(output_path, c='copy', an=None)
                .run(overwrite_output=True)
            )
            print(f"Final video saved to {output_path}")
            
            # Verify final video
            try:
                final_probe = ffmpeg.probe(output_path)
                final_duration = float(final_probe['format']['duration'])
                final_video_info = next((stream for stream in final_probe['streams'] if stream['codec_type'] == 'video'), None)
                if final_video_info:
                    print(f"Final video: duration={final_duration}s, resolution={final_video_info['width']}x{final_video_info['height']}")
                else:
                    print("Warning: Final video has no video stream")
            except ffmpeg.Error as e:
                print(f"Error probing final video: {e.stderr.decode()}")
                
        except ffmpeg.Error as e:
            print(f"Error concatenating segments: {e.stderr.decode()}")
        
        # Clean up temporary files
        os.remove(concat_list_path)
        shutil.rmtree(temp_dir)
    else:
        print("No valid segments to process.")

# Example usage
if __name__ == "__main__":
    with open("config.json", "r") as cfg_file:
        config = json.load(cfg_file)
    video_path = config["output_file"]
    timestamps_path = config["output_segments_file"]
    output_path = config["output_video_9x16"]
    process_video(video_path, timestamps_path, output_path)