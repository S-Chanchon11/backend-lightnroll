import json
import pickle
import numpy as np
import csv
import matplotlib.pyplot as plt

class Utilities:

    def prepare_data(self,path):
        """Loads training dataset from json file.
            :param data_path (str): Path to json file containing data
            :return X (ndarray): Inputs
            :return y (ndarray): Targets
        """

        with open(path, "r") as fp:
            data = json.load(fp)

        X = np.array(data["pitch"])
        y = np.array(data["labels"])
        z = np.array(data["mapping"])

        
        return X, y, z
    
    def prepare_data_for_cnn(self,path):
        """Loads training dataset from json file.
            :param data_path (str): Path to json file containing data
            :return X (ndarray): Inputs
            :return y (ndarray): Targets
        """

        with open(path, "r") as fp:
            data = json.load(fp)

        X = np.array(data["pitch"])
        y = np.array(data["labels"])
        z = np.array(data["mapping"])

        X = X[..., np.newaxis]
        
        return X, y, z
    
    def load_data_csv(data_path):

        with open(data_path, "r") as fp:
            reader = csv.reader(fp)

            # Skip the header row if it exists
            next(reader, None)
            
            data = {}
            data["mapping"] = []
            data["pitch"] = []
            data["labels"] = []
            data["order"] = []

            for row in reader:
                data["mapping"].append(row[0])
                data["order"].append(row[1])

                # Extract individual pitch values from the string, convert to floats
                pitch_values = [float(x) for x in row[2].strip("[]").split(", ")]
                data["pitch"].append(pitch_values)
                data["labels"].append(int(row[3]))

        print(np.array(data["pitch"]))
        return data
    
    def train(model, epochs, batch_size, X_train, y_train):
        """Trains model
        :param epochs (int): Num training epochs
        :param batch_size (int): Samples per batch
        :param patience (int): Num epochs to wait before early stop, if there isn't an improvement on accuracy
        :param X_train (ndarray): Inputs for the train set
        :param y_train (ndarray): Targets for the train set
        :param X_validation (ndarray): Inputs for the validation set
        :param y_validation (ndarray): Targets for the validation set

        :return history: Training history
        """
        
        # train model
        history = model.fit(X_train,
                            y_train,
                            epochs=epochs,
                            batch_size=batch_size)
        
        return history
    

    def plot_history(history):
        """Plots accuracy/loss for training/validation set as a function of the epochs

        :param history: Training history of model
        :return:
        """

        fig, axs = plt.subplots(2)

        # create accuracy subplot
        axs[0].plot(history.history["accuracy"], label="accuracy")
        axs[0].plot(history.history['val_accuracy'], label="val_accuracy")
        axs[0].set_ylabel("Accuracy")
        axs[0].legend(loc="lower right")
        axs[0].set_title("Accuracy evaluation")

        # create loss subplot
        axs[1].plot(history.history["loss"], label="loss")
        axs[1].plot(history.history['val_loss'], label="val_loss")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Loss")
        axs[1].legend(loc="upper right")
        axs[1].set_title("Loss evaluation")

        plt.show()

    def save_model(self,model,path):

        with open(path, 'wb') as file:  
            pickle.dump(model, file)

        file.close()

    def predict_chord(self,model, X_test, y_test, z):

        y_pred = model.predict(X_test)
        print("\nKNN:")
        for i in range(len(X_test)):
            print(z[y_pred[i]])