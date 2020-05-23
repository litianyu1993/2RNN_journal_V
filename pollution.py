from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import numpy as np
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
# from keras.models import Sequential
# from keras.layers import Dense, Activation, LSTM
from sklearn.utils import check_array
from matplotlib import pyplot as plt

def mean_absolute_percentage_error(y_true, y_pred):
    #y_true, y_pred = check_array(y_true, y_pred)

    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true):
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# convert series to supervised learning
# def convert_to_supervised(data, length = 5):
#     dim = 1 if type(data) is list else data.shape[1]
#     exp = []
#     for i in range(data.shape[0]):

def series_to_supervised(data, n_in=5, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def prepare_data(add_bias = True, normalize= False, diff = False):
    dataset = read_csv('pollution.csv', header=0, index_col=0)

    dataset.dropna(subset = ["pm2.5"], inplace=True)
    #print(dataset['pm2.5'][0:5])
    values = dataset.values
    values = np.delete(values, 8, axis=1)

    # values = np.delete(values, 4, axis = 1)

    # normalize features
    # scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = StandardScaler()
    #print(values[:, 4][0:5])
    scaled = scaler.fit_transform(values)

    if not normalize:
        scaled = scaler.inverse_transform(scaled)

    if diff:
        scaled = difference_axis0(scaled)
    # plt.plot(scaled[0:100, 4])
    # plt.show()
    #scaled = values
    #print(scaled.shape)
    #print(scaler.mean_[4], scaler.var_[4])
    if add_bias:
        scaled = np.insert(scaled, scaled.shape[1], 1, axis = 1)
        #print(scaled[0:5])
    return scaled, scaler

def generate_pollution_data(data, length, out_dim = 1, num_examples = None, diff = False):
    #length = length + out_dim
    scaled = data[:, 4].reshape(data.shape[0], -1)
    scaled = np.insert(scaled, scaled.shape[1], 1, axis=1)
    dim = scaled.shape[1]
    if diff:
        length += 1
    reframed = series_to_supervised(scaled, length, out_dim)

    if num_examples is None:
        num_examples = reframed.values.shape[0]
    Examples = reframed.values[:num_examples]
    values = Examples.reshape(num_examples, length+out_dim, dim)
    values = np.swapaxes(values, 1, 2)
    if diff:
        values = difference(values)
    print('here', values.shape)
    return values[:, :, :-out_dim], values[:, 0, -out_dim:].reshape(num_examples, -1)

def difference(X):
    new_X = np.ones([X.shape[0], X.shape[1], X.shape[2] - 1])
    for i in range(X.shape[2] - 1):
        new_X[:, :, i] = X[:, :, i+1]  - X[:, :, i]
    new_X[:, -1, :] = 1
    return new_X

def difference_axis0(X):
    new_X = np.ones([X.shape[0]-1, X.shape[1]])
    for i in range(X.shape[0]-1):
        new_X[i, :] = X[i+1, :] - X[i, :]
    return new_X

def other_methods_prepare_data(n = None, l = 3, out_dim = 1, add_bias = True, normalize = True, diff = False):

    data, scaler = prepare_data(add_bias, normalize, diff)
    training_data = data[:-1000]
    test_data = data[-1000:]
    data_function = lambda l, n: generate_pollution_data(training_data, l, out_dim= out_dim, num_examples=n)
    data_function_test = lambda l: generate_pollution_data(test_data, l, out_dim= out_dim)
    Xtest, ytest = data_function_test(l=l*2)
    Xtrain, ytrain = data_function(n = n, l=l*2)
    return Xtrain, ytrain, Xtest, ytest, scaler

def linear_regression(Xtrain, ytrain, Xtest, ytest, scaler):
    regr = linear_model.LinearRegression()
    # Xtrain = Xtrain[:3000]
    # ytrain = ytrain[:3000]
    regr.fit(Xtrain, ytrain)

    y_pred = regr.predict(Xtest)
    ytest = ytest * (scaler.var_[4] ** 0.5) + scaler.mean_[4]
    y_pred = y_pred * (scaler.var_[4] ** 0.5) + scaler.mean_[4]
    print(y_pred[0:5])
    print(ytest[0:5])
    print('RMSE:', np.sqrt(np.mean((ytest - y_pred)**2)))
    print('mean_absolute_error: %.2f'
          % mean_absolute_error(ytest, y_pred))
    print('MAPE:', mean_absolute_percentage_error(ytest, y_pred))

def fully_connected_nn(Xtrain, ytrain, Xtest, ytest, scaler):
    model = Sequential()
    model.add(Dense(Xtrain.shape[1]*2, input_dim=Xtrain.shape[1]))
    model.add(Activation('relu'))
    model.add(Dense(ytrain.shape[1]))
    model.compile(optimizer='Adam',
                  loss='mse')
    model.fit(Xtrain, ytrain, epochs=100, verbose=False, batch_size=256)
    y_pred = model.predict(Xtest)
    ytest = ytest * (scaler.var_[4] ** 0.5) + scaler.mean_[4]
    y_pred = y_pred * (scaler.var_[4] ** 0.5) + scaler.mean_[4]
    print(scaler.var_[4])
    print(y_pred[0:5])
    print(ytest[0:5])
    print('RMSE:', np.mean((ytest - y_pred) ** 2) ** 0.5)
    print('mean_absolute_error: %.2f'
          % mean_absolute_error(ytest, y_pred))
    print('MAPE:', mean_absolute_percentage_error(ytest, y_pred))

def LSTM_RNN(Xtrain, ytrain, Xtest, ytest, scaler):
    Xtrain = np.swapaxes(Xtrain, 1,2)
    Xtest = np.swapaxes(Xtest, 1, 2)
    model = Sequential()
    model.add(LSTM(20, input_length=Xtrain.shape[1], input_dim=Xtrain.shape[2]))
    #model.add(LSTM(10, input_dim))
    # output shape: (1, 1)
    model.add(Dense(ytrain.shape[1]))
    model.compile(loss="mse", optimizer="adam")
    model.summary()
    model.fit(Xtrain, ytrain, epochs=1000, verbose=False, batch_size=256, validation_data=(Xtest, ytest))
    #Xtest = np.swapaxes(Xtest, 1, 2)
    #print(Xtest.shape, Xtrain.shape)
    y_pred = model.predict(Xtest)
    ytest = ytest * (scaler.var_[4] ** 0.5) + scaler.mean_[4]
    y_pred = y_pred * (scaler.var_[4] ** 0.5) + scaler.mean_[4]
    print(scaler.var_[4])
    print(y_pred[0:5])
    print(ytest[0:5])
    print('RMSE:', np.mean((ytest - y_pred) ** 2) ** 0.5)
    print('mean_absolute_error: %.2f'
          % mean_absolute_error(ytest, y_pred))
    print('MAPE:', mean_absolute_percentage_error(ytest, y_pred))


# Xtrain, ytrain, Xtest, ytest, scaler= other_methods_prepare_data(l = 3, add_bias= False, diff  = True)
# LSTM_RNN(Xtrain, ytrain, Xtest, ytest, scaler)
# Xtrain = Xtrain.reshape(Xtrain.shape[0], -1)
# Xtest = Xtest.reshape(Xtest.shape[0], -1)
# print(Xtrain.shape)
# linear_regression(Xtrain, ytrain, Xtest, ytest, scaler)
# fully_connected_nn(Xtrain, ytrain, Xtest, ytest, scaler)
