import pickle
from ModelMaster import KNN, SVM
from CNN import CNN
from Utilities import Utilities

if __name__ == "__main__":

    utils = Utilities()
    cnn = CNN()
    knn = KNN()
    svm = SVM()

    DATA_PATH = "data_enhance_ver_1.json"

    X_train, y_train, z_train = utils.prepare_data(DATA_PATH)
   
    knn.KNN(
        save_path="knn_model.h5",
        X=X_train,
        y=y_train,
        n_neighbors=5
    )

    # X_train, y_train, z_train = utils.prepare_data_for_cnn(DATA_PATH)
    # cnn.CNN(
    #     save_path="cnn_model.h5",
    #     X=X_train,
    #     y=y_train,
    #     epochs=30,
    #     batch_size=32,
    #     patience=5,
    #     learning_rate=0.001
    # )

    KNN_MODEL = "knn_model.h5"
    CNN_MODEL = "cnn_model.h5"

    TEST_PATH = "test_Am.json"
    X_test, y_test, z_test = utils.prepare_data(TEST_PATH)

    svm.SVM(
        X=X_train,
        y=y_train,
        z=z_train,
        X_test=X_test
    )

    knn_model = pickle.load(open(KNN_MODEL, 'rb'))
    cnn_model = pickle.load(open(CNN_MODEL, 'rb'))

    utils.predict_chord(knn_model,X_test=X_test,y_test=y_test,z=z_train)

    X_test, y_test, z_test = utils.prepare_data_for_cnn(TEST_PATH)
    
    cnn.predict_chord(cnn_model,X_test=X_test)
    


    
