# all imports
from flask import Flask, request, json, jsonify, make_response
from sklearn.model_selection import GridSearchCV, KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import sys
import pandas as pd
import numpy as np
import sklearn as sk
import keras

app = Flask(__name__)

# when the a specific url gets reached, with an POST method, wanting to give information
@app.route('/info', methods=['POST'])
def parse_request():
    # get the data
    json_data = request.get_json()
    # if the data is empty dont do anything
    if json_data == None:
        return {}
    # if there is data, turn it into a dictionary, enter it into the machine, and receive the result
    else:
        loaded_json = json.loads(json.dumps(json_data))
        new_prediction = grid.predict(scaler.transform(np.array([[float(loaded_json["Pregnancies"]),
                                                                  float(loaded_json["Glucose"]), float(loaded_json["BloodPres"]),
                                                                  float(loaded_json["Skinthick"]), float(loaded_json["Insulin"]),
                                                                  float(loaded_json["BMI"]), float(loaded_json["DiabetesPedi"]),
                                                                  float(loaded_json["Age"])]])))
        new_prediction = (new_prediction > 0.5)
        # return a json file with the result
        if new_prediction:
            return make_response(jsonify({"result": "True"}))
        else:
            return make_response(jsonify({"result": "False"}))


def create_model(neuron1, neuron2):
        # create model
        model = Sequential()
        model.add(Dense(8, input_dim = 8, kernel_initializer= 'normal', activation= 'linear'))
        model.add(Dense(2, input_dim = 8, kernel_initializer= 'normal', activation= 'linear'))
        model.add(Dense(1, activation='sigmoid'))

        # Compile the model
        adam = Adam(lr = 0.001)
        model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
        return model



if __name__ == '__main__':
    df = pd.read_csv("diabetes.csv")
    df.drop(df[df['Insulin'] < 1].index, inplace = True)
    # Convert df into a numpy array
    dataset = df.values
    print(dataset.shape)
    # Split dataset into input (x) and an output (Y)
    X = dataset[:,0:8]
    Y = dataset[:,8].astype(int)
    print(X.shape)
    print(Y.shape)
    print(Y[:5])
    # Normalize data to have a mean = 0, stdev = 1; remove any added biases (parameters won't be weighted unequally)
    scaler = StandardScaler().fit(X)
    print(scaler)
    # Transform and display the training data
    X_standardized = scaler.transform(X)
    data = pd.DataFrame(X_standardized)
    data.describe()
    # Define a random seed
    seed = 6
    np.random.seed(seed)
    # Create the model
    model = KerasClassifier(build_fn = create_model, epochs = 100, batch_size = 20, verbose = 0)

    # Define the grid search parameters
    neuron1 = [4, 8, 16]
    neuron2 = [2, 4, 8]

    # Make a dictionary of the grid search parameters
    param_grid = dict(neuron1=neuron1, neuron2=neuron2)

    # Build and fit the GridSearchCV
    grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = KFold(random_state=seed), refit = True, verbose = 10)
    grid_results = grid.fit(X_standardized, Y)

    # Summarize the best results
    print("Best: {0}, using {1}".format(grid_results.best_score_, grid_results.best_params_))
    means = grid_results.cv_results_['mean_test_score']
    stds = grid_results.cv_results_['std_test_score']
    params = grid_results.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print('{0} ({1}) with: {2}'.format(mean, stdev, param))


    # Generate new predictions with hyperparameters
    y_pred = grid.predict(X_standardized)
    print(y_pred.shape)
    print(y_pred[:5])

    # Generate a classsification report

    print(accuracy_score(Y, y_pred))
    print(classification_report(Y, y_pred))

    # Example prediction datapoint
    example = df.iloc[7]
    print(example)

    # Make a prediction using optimized deep neural network
    prediction = grid.predict(X_standardized[7].reshape(1, -1))
    print(prediction)

    # Predicting the Test set results
    y_pred = grid.predict(X_standardized)
    y_pred = (y_pred > 0.5)
    app.run()
