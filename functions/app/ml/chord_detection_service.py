import librosa
import tensorflow as tf
import numpy as np
import pickle

SAMPLES_TO_CONSIDER = 22050
MODEL_PATH = "model.keras"

class _ChordDection:
    model = None
    _mapping = [
        "Am",
        "Bb",
        "Em",
        "G",
        "F",
        "Dm",
        "C",
        "Bdim"
    ]
    _instance = None

    def predict(self, file_path):
            """

            :param file_path (str): Path to audio file to predict
            :return predicted_chord (str): Keyword predicted by the model
            """

            # extract MFCC
            MFCCs = self.preprocess(file_path)

            # we need a 4-dim array to feed to the model for prediction: (# samples, # time steps, # coefficients, 1)
            MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

            # get the predicted label
            predictions = self.model.predict(MFCCs)
            predicted_index = np.argmax(predictions)
            predicted_chord = self._mapping[predicted_index]
            return predicted_chord


    def preprocess(self, file_path, num_mfcc=13, n_fft=2048, hop_length=512):
        """Extract MFCCs from audio file.

        :param file_path (str): Path of audio file
        :param num_mfcc (int): # of coefficients to extract
        :param n_fft (int): Interval we consider to apply STFT. Measured in # of samples
        :param hop_length (int): Sliding window for STFT. Measured in # of samples

        :return MFCCs (ndarray): 2-dim array with MFCC data of shape (# time steps, # coefficients)
        """

        # load audio file
        signal, sample_rate = librosa.load(file_path)

        if len(signal) >= SAMPLES_TO_CONSIDER:
            # ensure consistency of the length of the signal
            signal = signal[:SAMPLES_TO_CONSIDER]

            # extract MFCCs
            MFCCs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                         hop_length=hop_length)
        return MFCCs.T


def Chord_Detection_Service():
    """Factory function for Keyword_Spotting_Service class.

    :return _Keyword_Spotting_Service._instance (_Keyword_Spotting_Service):
    """

    # ensure an instance is created only the first time the factory function is called
    if _ChordDection._instance is None:
        _ChordDection._instance = _ChordDection()
        _ChordDection.model = pickle.load(open(MODEL_PATH, 'rb'))
        #_ChordDection.model = tf.keras.models.load_model(MODEL_PATH)
    return _ChordDection._instance


if __name__ == "__main__":

    # create 2 instances of the keyword spotting service
    cds = Chord_Detection_Service()
    cds1 = Chord_Detection_Service()

    # check that different instances of the keyword spotting service point back to the same object (singleton)
    assert cds is cds1

    # make a prediction
    chord = cds.predict("/Users/snow/backend-lightnroll/functions/app/ml/guitar_chord/c_maj.wav")
    print(f"Recognize : {chord}")