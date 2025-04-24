import librosa
import numpy as np
import matplotlib.pyplot as plt

def get_music_pace(filename, window_size=5.0, hop_size=2.5):
    y, sr = librosa.load(filename)
    tempo_over_time = []

    duration = librosa.get_duration(y=y, sr=sr)
    num_windows = int((duration - window_size) / hop_size) + 1

    for i in range(num_windows):
        start = int(i * hop_size * sr)
        end = int(start + window_size * sr)
        y_window = y[start:end]

        tempo, _ = librosa.beat.beat_track(y=y_window, sr=sr)
        tempo_over_time.append((i * hop_size, tempo))

    # Normalize tempo to pace (1-10)
    tempos = [tempo for _, tempo in tempo_over_time]
    min_tempo = min(tempos)
    max_tempo = max(tempos)
    pace = [int(np.clip(1 + 9 * (t - min_tempo) / (max_tempo - min_tempo), 1, 10)) for t in tempos]

    return tempo_over_time, pace

def plot_pace(tempo_over_time, pace):
    times = [t for t, _ in tempo_over_time]

    plt.figure(figsize=(12, 5))
    plt.plot(times, pace, marker='o', linestyle='-', color='teal')
    plt.ylim(0, 11)
    plt.xlabel('Time (s)')
    plt.ylabel('Pace (1=slowest, 10=fastest)')
    plt.title('Music Pace Over Time')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    file_path = "cyhtm.mp3"  # Replace with your actual file
    tempo_over_time, pace = get_music_pace(file_path)
    plot_pace(tempo_over_time, pace)
