import librosa
import os
import numpy as np
from tuner.Tuner import Tuner
from utils.LoggerUtils import LogUtils


log = LogUtils()
tuner = Tuner()

class PreAudioProcess:
    

    def pitch_detector(self, hz):

        detected_note = tuner.frequency_to_note(frequency=hz)

        return detected_note


    def pitch_detector_demo(self):

        note_freq_set = [330.0, 247.0, 196.0, 147.0, 110.0, 82.0]

        note_list = [tuner.frequency_to_note(frequency=x) for x in note_freq_set]

        return note_list

    def getCurrentDirectory(self) -> str:
        dir = os.path.dirname(os.path.realpath(__file__))
        return dir

    # get time series of fundamental frequencies in Hertz.
    def pitch_recognition(self):
        # :-------- Warning! ---------:
        # .load() not support .m4a file
        # must convert into .wav file

        file_path = self.getCurrentDirectory()
        filename = "\c_maj.wav"
        y, sr = librosa.load(file_path + filename)

        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
        )
        # times = librosa.times_like(f0)

        # note = str(type(f0[0]))
        toFloat = f0.tolist()
        index = toFloat[270]
        note = librosa.hz_to_note(index)
        # print(len(toFloat))
        print("(Hz,Note):", index, note)

    

    def generate_guitar_chord(self, note, fret=0):
        # Define the standard tuning for a guitar
        tuning = ["E2", "A2", "D3", "G3", "B3", "E4"]

        # Find the index of the string that is closest to the given note
        closest_string_index = min(
            range(len(tuning)),
            key=lambda i: abs(librosa.note_to_hz(tuning[i]) - librosa.note_to_hz(note)),
        )

        # Calculate the frequency of the note on the selected string and fret
        string_frequency = librosa.note_to_hz(tuning[closest_string_index])
        fret_frequency = string_frequency * 2 ** (
            fret / 12
        )  # Assuming equal temperament

        # Create the chord by adding the frequencies for each string
        chord = [
            librosa.hz_to_note(fret_frequency * 2 ** (i / 12))
            for i in range(len(tuning))
        ]

        return chord

    def dynamics_recognition(self):
        y, sr = librosa.load(librosa.ex("trumpet"))

        D = np.abs(librosa.stft(y))
        amp = librosa.amplitude_to_db(D, ref=np.max)
        print(amp)

    def detect_chords(self):
        file_path = self.getCurrentDirectory()
        audio_file = file_path + "\d_maj_2.mp3"

        # Load audio file
        y, sr = librosa.load(audio_file)

        # Extract chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        print(chroma)
        # Sum the chroma features over time
        chroma_sum = np.sum(chroma, axis=1)
        print(chroma_sum)
        # Find the most prominent pitch class (chord)
        detected_chord_index = np.argmax(chroma_sum)
        print(detected_chord_index)
        print()
        # Map index to chord names (adjust as needed)
        chord_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        detected_chord = chord_names[detected_chord_index]

        return detected_chord


    def audio_to_Hertz(self, freq):
        f0 = librosa.pyin(
            freq[0], fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
        )
        toFloat = f0[0].tolist()
        index = toFloat[10]
        print("Hz :", round(index, 4))
        note = librosa.hz_to_note(index)

        return note

    def load_audio_file(self):
        file_path = self.getCurrentDirectory()
        audio_file = file_path + self.filename

        y, sr = librosa.load(audio_file)

        return y, sr


def main():

    prep_audio = PreAudioProcess()

    note = prep_audio.pitch_detector(hz=329.0) 

    print(note)
    #pre_ap = PreAudioProcess(filename="\c_maj.wav")

    
    # time_series = pre_ap.load_audio_file()

    # hz = pre_ap.audio_to_Hertz(freq=time_series)

    
    # hz = pre_ap.frequency_to_note(frequency=82.1)

    # app.loading_process()
    # pap.pitch_recognition()
    # pap.dynamics_recognition()

    # --- chord detection from GPT ---

    # note = pap.frequency_to_note(65.40639132514966)
    # chord = pap.generate_guitar_chord(note, fret=0)
    # print("Note:", note)
    # print("Guitar Chord:", chord)

    # --- ML model ---
    # pap.chord_detection_model()

    # --- Features Extraction ---

    # temp = pap.detect_chords()
    # print(temp)
    # tun = pap.tuning()
    # print(tun)


if __name__ == "__main__":
    main()
