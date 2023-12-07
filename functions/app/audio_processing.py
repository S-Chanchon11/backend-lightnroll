import librosa
import sys
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

    def pitch_recognition(self):
        y, sr = librosa.load(librosa.ex('trumpet'))
        f0, voiced_flag, voiced_probs = librosa.pyin(y,
                                                     fmin=librosa.note_to_hz('C2'),
                                                     fmax=librosa.note_to_hz('C7'))
        times = librosa.times_like(f0)
        #print(times)
        print(f0)


def main():

    pap = PreAudioProcess()
    #app.loading_process()
    #pap.pitch_recognition()

if __name__ == '__main__':
    main()