import os

# List of folders to clean
folders = [
    'input_media',
    'narration',
    'output_videos'
]

for folder in folders:
    if os.path.isdir(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")
    else:
        print(f"Not a directory: {folder}")
