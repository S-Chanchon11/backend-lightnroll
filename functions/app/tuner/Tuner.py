import librosa


class Tuner:

    def frequency_to_note(self, frequency):
        
        note, _ = librosa.core.hz_to_note(frequency)
        
        return note
    
