import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle

from Utilities import Utilities

utils = Utilities()

class CNN:
    
    def build_cnn(self,input_shape,learning_rate):
    
        model = tf.keras.models.Sequential()

        # 1st conv layer
        model.add(tf.keras.layers.Conv2D(filters=64, 
                                        kernel_size=(3, 3), 
                                        activation='relu', 
                                        input_shape=input_shape,
                                        kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                        padding='same'
                                        ))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(
            pool_size=(3, 3), 
            strides=(2,2), 
            padding='same'))

        # 2nd conv layer
        model.add(tf.keras.layers.Conv2D(filters=32,
                                        kernel_size=(3, 3),
                                        activation='relu',
                                        kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                        padding='same'
                                        ))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(
            pool_size=(3, 3), 
            strides=(2,2), 
            padding='same'))

        # 3rd conv layer
        model.add(tf.keras.layers.Conv2D(filters=32,
                                        kernel_size=(2, 2),
                                        activation='relu',
                                        kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                        padding='same'
                                        ))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2), 
            strides=(2,2), 
            padding='same'))

        # flatten output and feed into dense layer
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        tf.keras.layers.Dropout(0.3)

        # softmax output layer
        model.add(tf.keras.layers.Dense(8, activation='softmax'))

        loss="sparse_categorical_crossentropy"

        optimiser = tf.optimizers.legacy.Adam(learning_rate=learning_rate)

        # compile model
        model.compile(optimizer=optimiser,
                    loss=loss,
                    metrics=["accuracy"]
                    )

        # print model parameters on console
        model.summary()

        return model
    
    
    
    def CNN(self,save_path,X,y,epochs,batch_size,patience,learning_rate,validation_size=0):

        train_shape = (X.shape[1], X.shape[2], 1)
        # X_temp, X_validation, y_temp, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

        earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor="accuracy", min_delta=0.001, patience=patience)
        
        # predict_model()
        model = self.build_cnn(input_shape=train_shape,learning_rate=learning_rate)

        model.fit(
            X,
            y,
            epochs=epochs,
            batch_size=batch_size,
            # validation_data=(X_validation, y_validation),
            callbacks=[earlystop_callback]
            )

        utils.save_model(model=model,path=save_path)



    def predict_chord(self,model, X_test):
    
        _mapping = [
            "Am",
            "Em",
            "G",
            "A",
            "Dm",
            "C",
            "D"
        ]

        # with open(MODEL_PATH,'rb') as file:
        # mymodel = pickle.load(open(model, 'rb'))
        predictions = model.predict(X_test)
        predicted_index = np.argmax(predictions)
        predicted_chord = _mapping[predicted_index]
        print("CNN:")
        print(predicted_chord)
