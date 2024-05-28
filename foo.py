

import pickle
import numpy as np


def predict():

    arr = [
        0.007898993340082395,
        0.0017041645921993189,
        0.005874897888617676,
        0.009716453369153688,
        0.30015358725923524,
        0.007981144601063053,
        0.012355310312647557,
        0.012748387192721007,
        0.5101886368126427,
        0.010403886792843578,
        0.003488380713845341,
        0.1174861571249483
    ]
    # X_test = request.get_json()
    # y_test = request.form.get('labels')
    # print(X_test["pitch"].shape())
    # print()
    X = np.array(arr).reshape(1, -1)
    # y = X.astype(float)
    # y = np.array(y_test)
    model = pickle.load(open('ann_i_v3.h5','rb'))

    mapping = ["Am", "Em", "G", "A", "F", "Dm", "C", "D", "E", "B"]
    # print(X)
    result = model.predict(X)

    # for i in range(len(X_test)):

    #     print(mapping[result[i]], end=" ")
    print(mapping[int(result)])

predict()

    # return jsonify({'placement':str(mapping[result])})