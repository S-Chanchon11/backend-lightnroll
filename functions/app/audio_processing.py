import librosa
import sys

import numpy as np

from utils.LoggerUtils import LogUtils
import mir_eval
import matplotlib.pyplot as plt

log = LogUtils()
class PreAudioProcess:

    def loading_process(self):
        filename = librosa.ex('trumpet')
        y, sr = librosa.load(filename, sr=11025)
        print(y)
        print(sr)

    # get time series of fundamental frequencies in Hertz.
    def pitch_recognition(self):

        # :-------- Warning! ---------:
        # .load() not support .m4a file

        y, sr = librosa.load('/Users/snow/backend-lightnroll/functions/app/dataset/c_maj.m4a')

        f0, voiced_flag, voiced_probs = librosa.pyin(y,
                                                     fmin=librosa.note_to_hz('C2'),
                                                     fmax=librosa.note_to_hz('C7'))
        # times = librosa.times_like(f0)
        
        
        # note = str(type(f0[0]))
        toFloat = f0.tolist()
        note = librosa.hz_to_note(toFloat[2])

        print(note)

    def dynamics_recognition(self):
        y, sr = librosa.load(librosa.ex('trumpet'))

        D = np.abs(librosa.stft(y))
        amp = librosa.amplitude_to_db(D, ref=np.max)
        print(amp)

def main():

    pap = PreAudioProcess()
    #app.loading_process()
    pap.pitch_recognition()
    #pap.dynamics_recognition()

if __name__ == '__main__':
    main()