"""
1.record
2.onset
3.segmentation
4.PCP
5.load model
6.predict

"""

import json
from math import log2
import pickle
import time
import librosa
import numpy as np
import soundfile as sf
import audiosegment
import warnings
from pydub import AudioSegment
warnings.filterwarnings("ignore")

def pcp(y,sr, fref=261.63):

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



def onset_filter(onset_array):
    onset_array  = onset_array.tolist()
    print(onset_array)
    size = len(onset_array)
    
    for i in range(size-1):
        # print(f"i={i}")
        if i==size:
            break
        else:
            if i==size-1:
                break
            diff = onset_array[i+1] - onset_array[i]
            if diff < 30:
                avg = (onset_array[i] + onset_array[i+1])/2
                onset_array.remove(onset_array[i])
                onset_array.remove(onset_array[i])
                size -= 1
                onset_array.insert(i,int(avg))
                # print(len(onset_array))

    return onset_array

def trim_audio(y,sr,cut_points):
    # load the audio file
    audio = audiosegment.from_numpy_array(y,sr)
    # q = BPM / x
    output_file_path = "output/output"
    cut_duration = 1.5 * 1000
    time_series = []
    # iterate over the list of time intervals
    for i, cut_points in enumerate(cut_points):
        start_time = int(cut_points * 1000)
        end_time = start_time + cut_duration
        end_time = min(end_time,len(audio))
        chunk  = audio[start_time:end_time]
        output_file_path_i = f"{output_file_path}_{i}.mp3"
        # output_file_path = f"out{}"
        chunk.export(output_file_path_i, format="mp3")
        time.sleep(3)
        
        # _sample_rate = chunk.frame_rate
        # print(_sample_rate)
        # time_series.append(chunk.to_numpy_array())
        # print(str(chunk.to_numpy_array()))
        _y,_sr = sf.read(output_file_path_i)
        time_series.append(_y)
        # time_series,_sample_rate = sf.read(chunk)
        # sample_rates = [segment.frame_rate for segment in trimmed]
        # construct the output file path
    
    return time_series, _sr

def get_onset_times(y,sr):
    
    o_env = librosa.onset.onset_strength(y=y, sr=sr)
    times = librosa.times_like(o_env, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=o_env,sr=sr)
    filtered_onset  = onset_filter(onset_frames)
    timeList = times[filtered_onset]

    return timeList

def predict_1(data):

    X = np.array(data).reshape(1, -1)

    model = pickle.load(open("functions/ann_i_v3.h5", "rb"))

    mapping = ["Am", "Em", "G", "A", "F", "Dm", "C", "D", "E", "B"]

    result = model.predict(X)

    return str(mapping[int(result)])
def predict_2(data):

    X = np.array(data).reshape(1, -1)

    model = pickle.load(open("functions/ann_ii_v3.h5", "rb"))

    mapping = ["Am", "Em", "G", "A", "F", "Dm", "C", "D", "E", "B"]

    result = model.predict(X)

    return str(mapping[int(result)])
def predict_3(data):

    X = np.array(data).reshape(1, -1)

    model = pickle.load(open("functions/ann_iii_v3.h5", "rb"))

    mapping = ["Am", "Em", "G", "A", "F", "Dm", "C", "D", "E", "B"]

    result = model.predict(X)

    return str(mapping[int(result)])

def check_result():
    y,sr = librosa.load('output/segment_10.wav')
    pcp_val = pcp(y=y,sr=sr)
    print(pcp_val,end='\n')
    # print(str(predict(data=pcp_val)))


def main():
    # check_result()
    # songname = 'All of me - final.mp3'  # Em C G D
    # # songname = 'Someone you loved - final.mp3' # C G Am F
    songname = 'Am_C_G_D.mp3'
    
    y, sr = sf.read(songname)
    # # audio_path = 'path/to/your/audiofile.wav'
    # # y, sr = librosa.load(audio_path, sr=None)

    # Detect onsets in the audio
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    # Duration of each segment in seconds
    segment_duration = 1.0
    segment_samples = int(segment_duration * sr)

    # List to store segments' audio time series and sample rate
    segments_info = []

    # Cut the audio segments based on the detected onsets
    for onset_time in onset_times:
        start_sample = int(onset_time * sr)
        end_sample = start_sample + segment_samples
        segment = y[start_sample:end_sample]
        
        # If the segment is shorter than the required duration, pad with zeros (silence)
        if len(segment) < segment_samples:
            segment = np.pad(segment, (0, segment_samples - len(segment)), 'constant')
        # print(segment)
        # Append the segment info (audio time series and sample rate) to the list
        segments_info.append(segment)
    # print(segments_info)

    # Save the audio segments
    for i, segment in enumerate(segments_info):
        output_path = f'output/segment_{i+1}.wav'
        sf.write(output_path, segment, sr)
    
   
    # y, sr = sf.read('Someone you loved - final.mp3')
    
    # jsonObj = {
    #     "y" : y.tolist(),
    #     "sr" : sr
    # }
    # # y,sr = getAudioFeature(result)
    # sr=int(sr)
    # y=np.array(y,dtype='float32')
    # cut_points = get_onset_times(y=y,sr=sr)
    # new_y, new_sr = trim_audio(y,sr,cut_points)

    prediction_result_A = []
    prediction_result_B = []
    prediction_result_C = []
    for i in range(len(segments_info)):
        # print(segments_info[i][0])
        new_pcp = pcp(y=segments_info[i],sr=44100)
        # print(new_pcp)
        pcpjson = {
            "pcp" : new_pcp
        }
        with open('pcp.jsonl', 'a') as ppp:
            ppp.write(json.dumps(pcpjson,separators=(',', ':')))
            ppp.write('\n')
        # print(new_pcp)
        prediction_result_A.append(str(predict_1(data=new_pcp)))
        prediction_result_B.append(str(predict_2(data=new_pcp)))
        prediction_result_C.append(str(predict_3(data=new_pcp)))

    jsonObj = {
        "prediction_A" : prediction_result_A,
        "prediction_B" : prediction_result_B,
        "prediction_C" : prediction_result_C
    }
    
    # result = model_predict(data)
    # print(prediction_result)
    with open(f'{songname}_prediction.json', 'w') as f:
        json.dump(jsonObj, f)

if __name__ == '__main__':
    main()