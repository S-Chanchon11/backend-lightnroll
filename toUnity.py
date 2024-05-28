"""
1.record
2.onset
3.segmentation
4.PCP
5.load model
6.predict

"""

from math import log2
import pickle
import librosa
import numpy as np


def pcp(audio_path, fref):

    y, sr = librosa.load(audio_path)
    fft_val = np.fft.fft(y)

    N = len(fft_val)

    def M(l, fs, fref):
        if l == 0:
            return -1
        return round(12 * log2((fs * l) / (N * fref))) % 12

    pcp = [0 for p in range(12)]
    for p in range(12):
        for l in range((N // 2) - 1):
            temp = M(l, fs=sr, fref=fref)

            if p == temp:
                h = abs(fft_val[l]) ** 2 
                pcp[p] += h

    pcp_norm = [0 for p in range(12)]
    for p in range(12):
        pcp_norm[p] = pcp[p] / sum(pcp)

    return list(pcp_norm)

def model_predict(arr):
    
    X = np.array(arr).reshape(1, -1)
    
    model = pickle.load(open('ann_i_v3.h5','rb'))

    mapping = ["Am", "Em", "G", "A", "F", "Dm", "C", "D", "E", "B"]

    result = model.predict(X)

    return mapping[int(result)]


def main():

    data = pcp()

    result = model_predict(data)


if __name__ == '__main__':
    main()