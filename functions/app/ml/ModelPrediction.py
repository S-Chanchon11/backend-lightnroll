import json
import pickle
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = "data_maj_chord_v1.json"
TEST_PATH = "test.json"
MODEL = 'model.sav'

def load_data(data_path):
    """Loads training dataset from json file.
        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["pitch"])
    y = np.array(data["labels"])
    z = np.array(data["mapping"])
    return X, y,z

def predict_model():

    X, y, z = load_data(DATA_PATH)

    X_test, y_test, z_test = load_data(TEST_PATH)

    model_svc_rbf=SVC(kernel='rbf')
    model_knn = KNeighborsClassifier(n_neighbors=3)
    model_ada=AdaBoostClassifier(n_estimators=200,learning_rate=2)
    model_dt=DecisionTreeClassifier()
    model_svc_lin=SVC(kernel='linear')
    model_svc_rbf=SVC(kernel='rbf')

    model_knn.fit(X, y)
    model_ada.fit(X, y)
    model_dt.fit(X, y)
    model_svc_lin.fit(X, y)
    model_svc_rbf.fit(X, y)
    y_pred_knn = model_knn.predict(X_test)
    y_pred_ada = model_ada.predict(X_test)
    y_pred_dt = model_dt.predict(X_test)
    y_pred_svm_lin = model_svc_lin.predict(X_test)
    y_pred_svm_rbf = model_svc_rbf.predict(X_test)
    print("KNN: ")
    for i in range(len(X_test)):
        print(z[y_pred_knn[i]],end=' ' )
    print("\nAdaboost: ")
    for i in range(len(X_test)):
        print(z[y_pred_ada[i]],end=' ' )
    print("\nDecision tree: ")
    for i in range(len(X_test)):
        print(z[y_pred_dt[i]],end=' ' )
    print("\nSVM rbf: ")
    for i in range(len(X_test)):
        print(z[y_pred_svm_rbf[i]],end=' ' )
    print("\nSVM linear: ")
    for i in range(len(X_test)):
        print(z[y_pred_svm_lin[i]],end=' ' )

def predict_SVM():
   
    X, y, z = load_data(DATA_PATH)
    X_test, Y_test, z_test = load_data(TEST_PATH)
    loaded_model = pickle.load(open(MODEL, 'rb'))
    result = loaded_model.predict(X_test)
    
    for i in range(len(X_test)):
        print(z[result[i]],end='')

def function_tester():

    X, y,z = load_data(DATA_PATH)
    print(z)

if __name__ == "__main__":
    # predict_model()
    predict_SVM()
    # function_tester()