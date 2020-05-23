######################################################
### Generating synthetic data with a 2-RNN dynamic ###
######################################################
import numpy as np
from LinRNN import LinRNN
import torch
#import tensorflow as tf
'''
Random 2RNN data generating function
'''
def generate_data(mdl, N_samples, seq_length,noise_variance=0.):
    X = []
    Y = []
    for i in range(N_samples):
        X.append(np.random.normal(0, 1, [seq_length, mdl.input_dim]))
        Y.append(mdl.predict(X[-1]) + np.random.normal(0, noise_variance))
    X = np.swapaxes(np.asarray(X), 1, 2)
    return np.asarray(X),np.asarray(Y).squeeze()

def generate_random_LinRNN(num_states, input_dim, output_dim,
                           alpha_variance = 1., Omega_variance = 1., A_variance = 1.):
    alpha = np.random.normal(0, alpha_variance, num_states)
    Omega = np.random.normal(0, Omega_variance, [num_states, output_dim])
    A = np.random.normal(0, A_variance, [num_states, input_dim, num_states])

    mdl = LinRNN(alpha,A,Omega)
    X,y = generate_data(mdl, 1000, 4, noise_variance=0.)
    mdl.alpha /= (np.mean(y**2)*10)
    return mdl
'''
Addition data generation
'''
def generate_data_simple_addition(num_examples, seq_length, n_dim = 1, noise_level = 0.):
    X = np.random.rand(num_examples, n_dim + 1, seq_length)
    X[:, -1, :] = np.ones((num_examples, seq_length))
    Y = np.sum(X[:, :-1, :], axis=2)
    Y = Y.reshape(num_examples, -1) + np.random.normal(0, noise_level, [num_examples, n_dim]).reshape(num_examples, n_dim)
    #X = np.swapaxes(X, 1, 2)
    return X, Y

'''
Wind data generation
'''

def generate_wind_speed(train_file_path, test_file_path, mean_window_size = 12):
    data1 = np.genfromtxt(train_file_path, max_rows=111659, delimiter=',')
    data1 = data1[:, 13]
    data2 = np.genfromtxt(train_file_path, skip_header=111659, delimiter=',')
    data2 = data2[:, 13]
    data3 = np.genfromtxt(test_file_path, delimiter=',')
    data3 = data3[:, 13]
    data = np.insert(data1, len(data1), data2)
    data = np.insert(data, len(data), data3)
    train_test_split = (len(data1) + len(data2)) / len(data)
    nan_count = 0
    for i in range(len(data)):
        if type(data[i]) is str:
            data[i] = float(data[i])
        if type(data[i]) is str:
            print(data[i])
        if np.isnan(data[i]):
            nan_count += 1
            temp_sum = []
            for j in range(1, 6):
                if not np.isnan(data[i - j]):
                    temp_sum.append(data[i - j])
                    break
            for j in range(1, 6):
                if not np.isnan(data[i + j]):
                    temp_sum.append(data[i + j])
                    break
            data[i] = np.mean(np.asarray(temp_sum))

    temp_data = []
    for i in range(int((len(data) - mean_window_size) / mean_window_size)):
        temp_data.append(np.mean(data[i * mean_window_size:(i * mean_window_size + mean_window_size)]))
    data = temp_data
    return data, train_test_split

def pad_data(X_vec, Y_vec, max_length = None):
    if max_length is None:
        max_length = X_vec[-1].shape[2]

    num_examples = 0
    for X in X_vec:
        num_examples += X.shape[0]
    #print('here', X_vec[0].shape)
    padded = np.zeros((num_examples, X_vec[0].shape[1]+1, max_length))
    #print('pad', padded.shape)
    if Y_vec[0].ndim == 1:
        out_dim = 1
    else:
        out_dim = Y_vec[0].shape[1]
    print('out', out_dim)
    y = np.zeros((num_examples, out_dim))

    current_pos = 0
    for X in X_vec:
        padded[current_pos:(current_pos + X.shape[0]), :X.shape[1], :X.shape[2]] = X
        if X.shape[2] < max_length:
            padded[current_pos:(current_pos + X.shape[0]), -1, X.shape[2]:] = 1.
        current_pos = current_pos + X.shape[0]
    current_pos = 0
    for Y in Y_vec:
        y[current_pos:(current_pos+Y.shape[0])] = Y
        current_pos = current_pos + X.shape[0]
    return padded, y



def generate_wind_speed_preprocess(data, length):
    X = []
    Y = []
    for i in range(len(data) - length - 1):
        temp = []
        for j in range(length):
            temp_data = np.asarray(data[i+j])
            temp.append(np.insert(temp_data, 0, 1.))
        temp = np.asarray(temp)
        X.append(temp)
        Y.append(np.asarray(data[i + length]))

    return np.asarray(X), np.asarray(Y)

def generate_wind_train(data, train_test_split, length):
    X, Y= generate_wind_speed_preprocess(data, length)
    train_X = X[:int(train_test_split * len(X))]
    train_Y = Y[:int(train_test_split * len(X))]
    return train_X, train_Y

def generate_wind_test(data, train_test_split, length):
    X, Y= generate_wind_speed_preprocess(data, length)
    test_X = X[int(train_test_split * len(X)):]
    test_Y = Y[int(train_test_split * len(X)):]
    return test_X, test_Y


def pred_k_more(model, test_X, test_Y, pred, ph, if_tc = False, if_tf = False):
    if ph <=0:return pred, test_Y

    for j in range(ph):
        temp_test_x = np.zeros((test_X.shape[0], test_X.shape[1]+1, test_X.shape[2]))

        for i in range(test_X.shape[0]):
            temp_test_x[i] = np.insert(test_X[i], test_X[i].shape[0]*test_X[i].shape[1],
                                       np.asarray([ 1., pred[i]])).reshape(test_X.shape[1]+1, test_X.shape[2])
            test_X[i] = temp_test_x[i][1:]
        pred2 = []
        if if_tc:
            Xtest_temp = torch.from_numpy(test_X).float()
            pred2 = model(Xtest_temp).detach().numpy()
        for i in range(len(test_X)):
            if if_tf == False and if_tc == False:
                pred2.append(model.predict(test_X[i]))
            elif if_tf == True:
                Xtest_temp = tf.convert_to_tensor(test_X[i], np.float32)
                pred2.append(model.predict(Xtest_temp))
        pred = np.asarray(pred2)
    return pred[:-(ph)], test_Y[(ph):]





