from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from Utilities import Utilities

DATA_PATH = "data_pcp.json"
utils = Utilities()

class KNN:

    def KNN(self,save_path,X,y,n_neighbors):

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        #Create a KNN Classifier
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    
        model.fit(X, y)

        utils.save_model(model=model,path=save_path)

class SVM:

    def SVM(self,X,y,z,X_test,save_path=''):

        svclassifier_lin = SVC(kernel='linear')
        svclassifier_rbf = SVC(kernel='rbf')

        # Train the model using the training sets
        svclassifier_lin.fit(X, y)
        svclassifier_rbf.fit(X, y)

        y_pred_lin = svclassifier_lin.predict(X_test)
        y_pred_rbf = svclassifier_rbf.predict(X_test)

        print("\nSVM linear: ")
        for i in range(len(X_test)):
            print(z[y_pred_lin[i]],end=' ' )
        print("\nSVM rbf: ")
        for i in range(len(X_test)):
            print(z[y_pred_rbf[i]],end=' ' )
        
        # utils.save_model(model=svclassifier_lin,path=save_path)




