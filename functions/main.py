# Welcome to Cloud Functions for Firebase for Python!
# To get started, simply uncomment the below code or create your own.
# Deploy with `firebase deploy`

# The Cloud Functions for Firebase SDK to create Cloud Functions and set up triggers.

import pickle
from aubio import source, onset
from flask import jsonify, request
import librosa
import numpy as np
from pydub import AudioSegment


from firebase_functions import firestore_fn, https_fn
# The Firebase Admin SDK to access Cloud Firestore.
from firebase_admin import initialize_app, firestore
import google.cloud.firestore
import firebase_admin
from firebase_admin import db
from firebase_admin import credentials, storage

from app.utils.CONSTANTS import CERTIFICATE, DB_URL
from app.tuner.Tuner import Tuner

cred = credentials.Certificate(
    CERTIFICATE
    )

firebase_admin.initialize_app(
    cred,
    {
        "databaseURL": DB_URL
    },
)

bucket = storage.bucket()
# storage = firebase

"""
To deploy serverless
0.firebase init functions
0.firebase emulators:start

1.firebase deploy --only functions

2.You will get URL(s) then continue insert query into parameter

"""


@https_fn.on_request()
def on_request_example(req: https_fn.Request) -> https_fn.Response:
    return https_fn.Response("Hello world!")

# gs://lightnroll-11.appspot.com/audio/out_1.wav


# @https_fn.on_request()
# def read_database_on_realtime_database(req: https_fn.Request) -> https_fn.Response:

#     ref = db.reference("/all-chords")
    
#     doc = ref.child('C').get()

#     return doc


# @POST 
# /function?freq=
# @https_fn.on_request()
# def pitch_detection(req: https_fn.Request) -> https_fn.Response:

#     hz = req.args.get("freq")
#     if hz is None:
#         return https_fn.Response("No parameter provided", status=400)
#     else:
#         tuner = Tuner()

#         hz = float(hz)

#         note = tuner.frequencyToNote(frequency=hz)

#         return https_fn.Response(note)

def get_wav():
    return "gs://lightnroll-11.appspot.com/audio/out_1.wav"

def extract_audio_features(audio_path):

    y, sr = librosa.load(audio_path)

    return y,sr

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

    print(f'File {file_path} uploaded to {destination_blob_name}.')
    print(f'Public URL: {blob.public_url}')


def get_onset_times(sample_rate,file_path):

    window_size = 1024  # FFT size
    hop_size = window_size // 4
    src_func = source(file_path, sample_rate, hop_size)
    sample_rate = src_func.samplerate
    onset_func = onset("default", window_size, hop_size)

    duration = float(src_func.duration) / src_func.samplerate

    onset_times = []  # seconds
    while True:  # read frames
        samples, num_frames_read = src_func()
        if onset_func(samples):
            onset_time = onset_func.get_last_s()
        if onset_time < duration:
            onset_times.append(onset_time)
        else:
            break
        if num_frames_read < hop_size:
            break

    return onset_times


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

    model = pickle.load(open('ann_i_v3.h5','rb'))

    mapping = ["Am", "Em", "G", "A", "F", "Dm", "C", "D", "E", "B"]

    result = model.predict(X)


    return jsonify({'chord':str(mapping[int(result)])})
    
        

    
        



