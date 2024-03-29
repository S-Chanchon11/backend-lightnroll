# Welcome to Cloud Functions for Firebase for Python!
# To get started, simply uncomment the below code or create your own.
# Deploy with `firebase deploy`

# The Cloud Functions for Firebase SDK to create Cloud Functions and set up triggers.
import json

from firebase_functions import firestore_fn, https_fn

# The Firebase Admin SDK to access Cloud Firestore.
from firebase_admin import initialize_app, firestore
import google.cloud.firestore
import firebase_admin
from firebase_admin import db
from firebase_admin import credentials
import logging

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


# @firestore_fn.on_document_created(document="messages/{pushId}")
# def makeuppercase(
#     event: firestore_fn.Event[firestore_fn.DocumentSnapshot | None],
#     isToggleEnable: bool,
# ) -> None:
#     """Listens for new documents to be added to /messages. If the document has
#     an "original" field, creates an "uppercase" field containg the contents of
#     "original" in upper case."""
#
#     isToggleEnable = False
#
#     if isToggleEnable:
#         # Get the value of "original" if it exists.
#         if event.data is None:
#             return
#         try:
#             original = event.data.get("original")
#         except KeyError:
#             # No "original" field, so do nothing.
#             return
#
#         # Set the "uppercase" field.
#         print(f"Uppercasing {event.params['pushId']}: {original}")
#         upper = original.upper()
#         event.data.reference.update({"uppercase": upper})
#     else:
#         return https_fn.Response(f"{__name__} is disabled", status=500)


# @https_fn.on_request()
# def upload_data_to_database(req: https_fn.Request) -> https_fn.Response:
#     # Grab the text parameter.
#     variant = req.args.get("variant")
#     if variant is None:
#         return https_fn.Response("No text parameter provided", status=400)

#     firestore_client: google.cloud.firestore.Client = firestore.client()

#     with open(variant + ".json", "r", encoding="utf-8") as output:
#         jsonObj = list(output)

#     for obj in jsonObj:
#         # Parse the JSON data
#         parsed_data = json.loads(obj)
#         # Access the values
#         for key, value in parsed_data.items():
#             for chord_data in value:
#                 pos = chord_data["positions"]
#                 finger = chord_data["fingerings"]
#                 _, doc_ref = firestore_client.collection("chord").add
#                 ({key: {"position": pos, "fingerings": [finger]}})
#     return https_fn.Response(f"Message with ID {doc_ref.id} added.")


# @https_fn.on_request()
# def read_database_on_cloud_function(req: https_fn.Request) -> https_fn.Response:
#     db: google.cloud.firestore.Client = firestore.client()

#     doc_ref = db.collection("chord-data").document("C-major")

#     doc = doc_ref.get()

#     if doc.exists:
#         print(f"Document data: {doc.to_dict()}")
#     else:
#         print("No such document!")

#     return https_fn.Response(f"Get Chord with ID {doc_ref.id}.")


@https_fn.on_request()
def read_database_on_realtime_database(req: https_fn.Request) -> https_fn.Response:

    ref = db.reference("/all-chords")
    
    doc = ref.child('C').get()

    return doc


# @POST 
# /function?freq=
@https_fn.on_request()
def pitch_detection(req: https_fn.Request) -> https_fn.Response:

    hz = req.args.get("freq")
    if hz is None:
        return https_fn.Response("No parameter provided", status=400)
    else:
        tuner = Tuner()

        hz = float(hz)

        note = tuner.frequencyToNote(frequency=hz)

        return https_fn.Response(note)
    
        

    
        



