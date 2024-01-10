import librosa


class Tuner:

    def frequencyToNote(self, frequency):
        
        note, _ = librosa.hz_to_note(frequency)
        
        return note
    
