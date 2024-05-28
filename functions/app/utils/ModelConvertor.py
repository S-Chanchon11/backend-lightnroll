import tensorflow as tf
from keras.models import load_model

def convert_model(saved_model_dir):
    
    # load = load_model(saved_model_dir)
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
    tflite_model = converter.convert()

    # Save the model.
    with open('ann1.tflite', 'wb') as f:
        f.write(tflite_model)


def main():
    
    convert_model(saved_model_dir="/Users/snow/ml-lightnroll/ann_i_v4.h5")





if __name__ == '__main__':
    main()