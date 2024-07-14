import os
import librosa
import numpy as np
from pydub import AudioSegment

def load_and_preprocess_audio(file_path, sr=44100):
    audio, fs = librosa.load(file_path, sr=sr)
    # Apply noise reduction or other preprocessing if needed
    return audio, fs

def detect_tempo(audio, fs=44100):
    tempo, _ = librosa.beat.beat_track(y=audio, sr=fs)
    return tempo

def evaluate_tempo(detected_tempo, expected_tempo, tolerance=5):
    if abs(detected_tempo - expected_tempo) <= tolerance:
        return "Playing at the correct tempo."
    elif detected_tempo > expected_tempo:
        return "Playing too fast."
    else:
        return "Playing too slow."

def process_audio_files(directory, expected_tempo):
    results = []
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory, filename)
            audio, fs = load_and_preprocess_audio(file_path)
            detected_tempo = detect_tempo(audio, fs)
            result = evaluate_tempo(detected_tempo, expected_tempo)
            results.append((filename, detected_tempo, result))

    return results

def chunk_audio(audio_file, time_intervals, output_folder):
    # Load the audio file
    audio = AudioSegment.from_file(audio_file)
    
    # Ensure output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop through the time intervals and create chunks
    for i, (start_time, end_time) in enumerate(time_intervals):
        # Convert times to milliseconds
        start_time_ms = max(0, (start_time - 0.5) * 1000)  # 1 second before
        end_time_ms = min(len(audio), (end_time) * 1000)  # 1 second after

        # Extract the chunk
        chunk = audio[start_time_ms:end_time_ms]
        
        # Export the chunk
        chunk.export(os.path.join(output_folder, f"chunk_{i}.wav"), format="wav")
        print(f"Exported chunk_{i}.wav from {start_time_ms}ms to {end_time_ms}ms")


if __name__ == "__main__":
    directory = "data"  # Directory containing recorded audio files
    expected_tempo = 75  # Example expected tempo in BPM
    results = process_audio_files(directory, expected_tempo)

    for filename, detected_tempo, result in results:
        print(f"File: {filename}")
        print(f"Detected Tempo: {detected_tempo} BPM")
        print(result)

    # audio_file = "data/record_all.wav"
    # time_intervals = [(0, 1.5), (3, 4.5), (6, 7.5),(9, 10.5), (12, 13.5), (15, 16.5),(18,19.5),(24,25)]  # List of (start_time, end_time) in seconds
    # output_folder = "chunks"

    # chunk_audio(audio_file, time_intervals, output_folder)