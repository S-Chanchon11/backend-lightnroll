# Welcome to Cloud Functions for Firebase for Python!
# To get started, simply uncomment the below code or create your own.
# Deploy with `firebase deploy`

# The Cloud Functions for Firebase SDK to create Cloud Functions and set up triggers.

import pickle
import sys
from aubio import source, onset
from flask import jsonify
import librosa
import numpy as np
from pydub import AudioSegment
import soundfile as sf
import io
import json
from firebase_functions import firestore_fn, https_fn

# The Firebase Admin SDK to access Cloud Firestore.
from firebase_admin import initialize_app, firestore
import google.cloud.firestore
import firebase_admin
from firebase_admin import db
from firebase_admin import credentials, storage

from app.utils.CONSTANTS import CERTIFICATE, DB_URL
from app.tuner.Tuner import Tuner

cred = credentials.Certificate(CERTIFICATE)

firebase_admin.initialize_app(
    cred, {"storageBucket": "lightnroll-11.appspot.com", "databaseURL": DB_URL}
)

bucket = storage.bucket()
# storage = firebase

"""
To deploy local
0.firebase init functions
0.firebase emulators:start

1.firebase deploy --only functions

2.You will get URL(s) then continue insert query into parameter

"""


@https_fn.on_request()
def on_request_example(req: https_fn.Request) -> https_fn.Response:
    return https_fn.Response("Hello world!")

# @https_fn.on_request()
# def read_database_on_realtime_database(req: https_fn.Request) -> https_fn.Response:

#     ref = db.reference("/all-chords")

#     doc = ref.child('C').get()

#     return doc

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
    
def get_wav_y(path):
    # http://127.0.0.1:5001/lightnroll-11/us-central1/get_wav?path=audio/out_1.wav
    # path = req.args.get("path")
    np.set_printoptions(threshold=sys.maxsize)
    
    if path is None:
        return https_fn.Response("No parameter provided", status=400)
    else:
        
        blob = bucket.blob(path)
        audio_bytes = blob.download_as_bytes()
        audio_buffer = io.BytesIO(audio_bytes)
        y, sr = librosa.load(audio_buffer)

        return y
def get_wav_sr(path):
    # http://127.0.0.1:5001/lightnroll-11/us-central1/get_wav?path=audio/out_1.wav
    # path = req.args.get("path")
    np.set_printoptions(threshold=sys.maxsize)
    
    if path is None:
        return https_fn.Response("No parameter provided", status=400)
    else:
        
        blob = bucket.blob(path)
        audio_bytes = blob.download_as_bytes()
        audio_buffer = io.BytesIO(audio_bytes)
        y, sr = librosa.load(audio_buffer)

        return sr
def extract_audio_features(buffer):
    y, sr = librosa.load(buffer)

    return y,sr

def onset_filter(onset_array):
    onset_array  = onset_array.tolist()
    size = len(onset_array)
    # print(size)
    for i in range(size-1):
        # print(i)
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

    
@https_fn.on_request()
def get_onset_times(req: https_fn.Request) -> https_fn.Response:

    path = req.args.get("path")
    buffer = get_wav_in_bytes(path)
    y, sr = extract_audio_features(buffer)
    o_env = librosa.onset.onset_strength(y=y, sr=sr)
    times = librosa.times_like(o_env, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=o_env,sr=sr)
    filtered_onset  = onset_filter(onset_frames)
    timeList = times[filtered_onset]

    return https_fn.Response(str(timeList))

@https_fn.on_request()
def upload_audio_to_firebase(file_path):
    """Uploads an audio file to Firebase Storage."""

    destination_blob_name = "gs://lightnroll-11.appspot.com/audio/"
    # Get a reference to the storage bucket
    # bucket = storage.bucket()

    # Create a new blob and upload the file's content
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(file_path)

    # Make the blob publicly viewable (optional)
    blob.make_public()

    print(f"File {file_path} uploaded to {destination_blob_name}.")
    print(f"Public URL: {blob.public_url}")





@https_fn.on_request()
def trim_audio(intervals, input_file_path, output_file_path="out"):
    # load the audio file
    audio = AudioSegment.from_file(input_file_path)

    # iterate over the list of time intervals
    for i, (start_time, end_time) in enumerate(intervals):
        # extract the segment of the audio
        segment = audio[start_time * 1000 : end_time * 1000]
        # construct the output file path
        output_file_path_i = f"{output_file_path}_{i}.wav"
        # export the segment to a file
        # segment.export(output_file_path_i, format="wav")
        url = "gs://lightnroll-11.appspot.com/audio/" + output_file_path_i


@https_fn.on_request()
def predict(req: https_fn.Request) -> https_fn.Response:

    X_test = req.get_json()

    X = np.array(X_test["pitch"]).reshape(1, -1)

    model = pickle.load(open("ann_i_v3.h5", "rb"))

    mapping = ["Am", "Em", "G", "A", "F", "Dm", "C", "D", "E", "B"]

    result = model.predict(X)

    return jsonify({"chord": str(mapping[int(result)])})
