# Welcome to Cloud Functions for Firebase for Python!
# To get started, simply uncomment the below code or create your own.
# Deploy with `firebase deploy`

# The Cloud Functions for Firebase SDK to create Cloud Functions and set up triggers.

import json
from math import log2
import math
import pickle
import sys
# from aubio import source, onset
from flask import jsonify
import librosa
import numpy as np
# from pydub import AudioSegment
# import soundfile as sf
import io
# import json
# import soundfile as sf
from firebase_functions import firestore_fn, https_fn
import audiosegment
# The Firebase Admin SDK to access Cloud Firestore.
# from firebase_admin import initialize_app, firestore
import firebase_admin
from firebase_admin import db, credentials, storage, firestore
import os
import librosa
import numpy as np
from pydub import AudioSegment
from urllib.parse import urlparse
from collections import Counter

from app.utils.CONSTANTS import CERTIFICATE, DB_URL
from app.tuner.Tuner import Tuner

cred = credentials.Certificate(CERTIFICATE)

firebase_admin.initialize_app(
    cred, 
    {"storageBucket": "lightnroll-11.appspot.com", 
     "databaseURL": DB_URL}
)


bucket = storage.bucket()
# storage = firebase

"""
To deploy local
0.firebase init functions
0.firebase emulators:start

1.firebase deploy --only functions

2.You will get URL(s) then continue insert query into parameter

http://127.0.0.1:5001/lightnroll-11/us-central1/main?path=audio/G_E_1.wav
"""
# cs = storage.storage
ref = db.reference("/all-chords")
getData = ref.get()
rt_ref = db.reference("/")
audio_feature_ref = db.reference("/audio_feature")
ml_ref = db.reference("/ml_predict")
# db_firestore = firestore.Client()
fs: firestore.Client = firestore.client()
users_ref = fs.collection("users")

@https_fn.on_request()
def on_request_example(req: https_fn.Request) -> https_fn.Response:
    return https_fn.Response("Hello world!")

@https_fn.on_request()
def get_pcp(req: https_fn.Request) -> https_fn.Response:
    pcp = [0.1,0.2,0.3,0.4,0.5]
    jsonObj = {
        "pcp" : pcp
    }
    return jsonify(jsonObj)

@https_fn.on_request()
def send_score(req: https_fn.Request) -> https_fn.Response:

    jsonObj = {
        "score":req.args.get("score")
    }
    users_ref = fs.collection("users")
    users_ref.document('DSPlOrYFN6d2e00lOlERTh3riTj1').update(jsonObj)

    return https_fn.Response("Success")

@https_fn.on_request()
def read_firestore(req: https_fn.Request) -> https_fn.Response:

    
    users_ref.document('DSPlOrYFN6d2e00lOlERTh3riTj1').get()
    query_docs = users_ref.stream()

    for doc in query_docs:
        print(f'{doc.id} => {doc.to_dict()}')

    return "Done"

@https_fn.on_request()
def read_database_on_realtime_database(req: https_fn.Request):
    get_new_ref = audio_feature_ref.get()
    # ref = db.reference("/all-chords")
    return get_new_ref


@https_fn.on_request()
def write(req: https_fn.Request):

   
    sr = "44100"
    # ref = db.reference('audio_feature')
    # user_ref = ref.child('_uid')
    jsonobj = {
        "audio_feature" : [
             {
                "uuid" : "19e7Ml7rNIXvJbl0QgIpNoyftUZ2",
                "sr" : sr,
                "y" : y
             }     
        ]
    }
    rt_ref.set(jsonobj)
    # ref = db.reference("/all-chords")
    return https_fn.Response("UPLOAD DONE")

def load_model():
    # storage_client = storage.Client()
    # bucket = storage.bucket()
    blob = bucket.blob("ann_i_v3.h5")
    
    # Download the model file into an in-memory bytes buffer
    model_bytes = blob.download_as_bytes()
    
    # Load the model using pickle
    model = pickle.loads(model_bytes)
    
    return model

def write_to_firestore(json_data):
    
    jsonObj = {
        "uuid" : "19e7Ml7rNIXvJbl0QgIpNoyftUZ2",
        "prediction" : str(json_data)
    }

    send_data = firestore_client.collection("prediction").document("a11")
    send_data.set(jsonObj)
    

def getAudioFeature_collection():
    
    # Fetch all users data
    users_data = audio_feature_ref.get()

    # Iterate through users to find the specific userId
    for user in users_data:
        # print(user)
        if user['uuid'] == "19e7Ml7rNIXvJbl0QgIpNoyftUZ2":
            return user

def getAudioFeature(data):
    # get_audio_feature = audio_feature_ref.get()
    y = data["y"]
    sr = data["sr"]
    # ref = db.reference("/all-chords")
    return y,sr
# @https_fn.on_request()
# def get_wav(req: https_fn.Request):

def get_wav_in_bytes(path):
    # http://127.0.0.1:5001/lightnroll-11/us-central1/get_wav?path=audio/out_1.wav
    # path = req.args.get("path")
    np.set_printoptions(threshold=sys.maxsize)
    
    if path is None:
        return https_fn.Response("No parameter provided", status=400)
    else:
        
        blob = bucket.blob(path)
        audio_bytes = blob.download_as_bytes()
        audio_buffer = io.BytesIO(audio_bytes)

        return audio_buffer

@https_fn.on_request()
def get_wav_audio_feature(req: https_fn.Request):
    # http://127.0.0.1:5001/lightnroll-11/us-central1/get_wav?path=audio/out_1.wav
    path = req.args.get("path")
    np.set_printoptions(threshold=sys.maxsize)
    
    if path is None:
        return https_fn.Response("No parameter provided", status=400)
    else:
        
        blob = bucket.blob(path)
        audio_bytes = blob.download_as_bytes()
        audio_buffer = io.BytesIO(audio_bytes)
        y, sr = librosa.load(audio_buffer)

        return sr
    
# def get_wav_sr(path):
#     # http://127.0.0.1:5001/lightnroll-11/us-central1/get_wav?path=audio/out_1.wav
#     # path = req.args.get("path")
#     np.set_printoptions(threshold=sys.maxsize)
    
#     if path is None:
#         return https_fn.Response("No parameter provided", status=400)
#     else:
        
#         blob = bucket.blob(path)
#         audio_bytes = blob.download_as_bytes()
#         audio_buffer = io.BytesIO(audio_bytes)
#         y, sr = librosa.load(audio_buffer)

#         return sr
def extract_audio_features(buffer):
    y, sr = librosa.load(buffer)
    # y, sr = sf.read(buffer)

    return y,sr

# def get_audio_feat():

#     get_new_ref = new_ref.get("sr")
#     # ref = db.reference("/all-chords")
#     return get_new_ref

def onset_filter(onset_array):
    onset_array  = onset_array.tolist()
    size = len(onset_array)
    for i in range(size-1):
        if onset_array[i]==onset_array[size-1]:
            break
        else:
            diff = onset_array[i+1] - onset_array[i]
            if diff < 10:
                avg = (onset_array[i] + onset_array[i+1])/2
                onset_array.remove(onset_array[i])
                onset_array.remove(onset_array[i])
                size -= 1
                onset_array.insert(i,int(avg))
    return onset_array

    
# @https_fn.on_request()
def get_onset_times(y,sr):
    
    o_env = librosa.onset.onset_strength(y=y, sr=sr)
    times = librosa.times_like(o_env, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=o_env,sr=sr)
    filtered_onset  = onset_filter(onset_frames)
    timeList = times[filtered_onset]

    return timeList

def onset_and_extract(y,sr):
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    # Duration of each segment in seconds
    segment_duration = 2.0
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
    
    return segments_info

def trim_audio(y,sr,cut_points):
    # load the audio file
    audio = audiosegment.from_numpy_array(y,sr)
    cut_duration = 1 * 1000
    time_series = []
    # iterate over the list of time intervals
    for i, cut_points in enumerate(cut_points):
        start_time = int(cut_points * 1000)
        end_time = start_time + cut_duration
        end_time = min(end_time,len(audio))
        chunk  = audio[start_time:end_time]
        _sample_rate = chunk.frame_rate
        time_series.append(chunk.to_numpy_array())
        # sample_rates = [segment.frame_rate for segment in trimmed]
        # construct the output file path
    
    return time_series, _sample_rate
        # output_file_path_i = f"{output_file_path}_{i}.wav"
        # export the segment to a file
        # chunk.export(output_file_path_i, format="wav")
        # url = "gs://lightnroll-11.appspot.com/audio/" + output_file_path_i



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

def write_audio_to_realtimeDB(result):

    ml_ref.set(result)
    # ref = db.reference("/all-chords")
    return https_fn.Response("UPLOAD DONE")

def load_and_preprocess_audio(file_path, sr=44100):
    audio, fs = librosa.load(file_path, sr=sr)
    # Apply noise reduction or other preprocessing if needed
    return audio, fs

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
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
    # for filename in os.listdir(directory):
    #     # print(filename)
    #     if filename.endswith(".wav"):
    #         file_path = os.path.join(directory, filename)
    audio, fs = load_and_preprocess_audio(directory)
    detected_tempo = detect_tempo(audio, fs)
    result = evaluate_tempo(detected_tempo, expected_tempo)
    # results.append((directory, detected_tempo, result))
    return detected_tempo

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
        # print(f"Exported chunk_{i}.wav from {start_time_ms}ms to {end_time_ms}ms")

def flatten_chord_list(nested_list):
    flattened_list = [chord for sublist in nested_list for chord in sublist]
    return flattened_list

@https_fn.on_request()
def preprocessing(req: https_fn.Request) -> https_fn.Response:
    params = req.get_json()

    directory = params.get("rid") # Directory containing recorded audio files
    print(directory)
    blob = bucket.blob(directory + '/' + "Recording.wav")
    
    local_file_path = directory+"_recording.wav"


    try:
        blob.download_to_filename(local_file_path)
        print("File downloaded successfully!")
    except Exception as e:
        print(f"Error downloading file: {e}")
    
    expected_tempo = params.get("tempo")  

    results = process_audio_files(local_file_path, expected_tempo)
    jsonObj = {}
    
    arr_list = results.tolist()

    # Serialize to JSON
    round_num = round(arr_list[0])
    json_data = json.dumps(round_num)
    
    jsonObj["detected_tempo"] = json_data
    print(jsonObj["detected_tempo"])
    # if(round_num > expected_tempo+10 or round_num < expected_tempo-10):
    #     print(json_data)
    #     jsonObj["detected_tempo"] = 0
    #     return jsonify(jsonObj)
    # else:

    time_intervals = [(0,1.5), (3,4.5), (6,7.5),(9.0,10.5),(12,13.5),(15,16.5),(18,19.5),(20,24.5)]  # List of (start_time, end_time) in seconds
    output_folder = "chunks"

    chunk_audio(local_file_path, time_intervals, output_folder)
    pcp_list = []
    prediction_result = []
    for dirpath, _, filenames in os.walk(output_folder):
        for filename in filenames:
            # Full path to the file
            file_path = os.path.join(dirpath, filename)
            # print(file_path)
            y,sr = librosa.load(file_path,sr=22050)

            pcpResult = pcp(y,sr)

            # print(pcpResult) 

            pcp_list.append(pcpResult)

    for index, layer in enumerate(pcp_list):
        result = predict(data=layer)
        prediction_result.append(result)
    
    jsonObj["pred_result"] = flatten_chord_list(prediction_result)


    return jsonify(jsonObj) 


@https_fn.on_request()
def evaluate(req: https_fn.Request) -> https_fn.Response:

    tmp = req.get_json()
    print(tmp)
    pcp_list = tmp.get("pcp_list", [])
    print(pcp_list)
    prediction_result = []
    for index, layer in enumerate(pcp_list):
        print(layer)
        result = predict(data=layer)
        prediction_result.append(result)
    

    foo = {
        "pred_result" : prediction_result
    }
    print(foo)

    return jsonify(foo)

# @https_fn.on_request()
def predict(data):
    if(math.isnan(data[0])):
        return ""
    else:
        X = np.array(data).reshape(1, -1)

        model1 = pickle.load(open("ann_i_v3.h5", "rb"))
        model2 = pickle.load(open("ann_ii_v3.h5", "rb"))
        model3 = pickle.load(open("ann_iii_v3.h5", "rb"))

        mapping = ["Am", "Em", "G", "A", "F", "Dm", "C", "D", "E", "B"]

        result1 = model1.predict(X)
        result2 = model2.predict(X)
        result3 = model3.predict(X)

        fromANN1 = str(mapping[int(result1)])
        fromANN2 = str(mapping[int(result2)])
        fromANN3 = str(mapping[int(result3)])

        combined_predictions = list(zip(fromANN1, fromANN2, fromANN3))

# Determine the majority prediction for each position
        final_result = [Counter(predictions).most_common(1)[0][0] for predictions in combined_predictions]
        

        return final_result

