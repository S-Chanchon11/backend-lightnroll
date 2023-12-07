import firebase_admin
from firebase_admin import db
from firebase_admin import credentials


def realtime_database():

    #default_app = firebase_admin.initialize_app()
    cred = credentials.Certificate("/Users/celeven/Documents/MUIC/lightnroll-11-firebase-adminsdk-ic5jr-e24b4b8f17.json")
    
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://lightnroll-11-default-rtdb.asia-southeast1.firebasedatabase.app'
    })
    ref = db.reference('/all-chords/')
    
    print(ref.child('C').get())



realtime_database()