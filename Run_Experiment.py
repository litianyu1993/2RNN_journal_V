import numpy as np
#import tt
import learning
import pickle
import os
import sys
from shutil import copyfile
import time
import pollution
import matplotlib.pyplot as plt
import argparse
import synthetic_data
from TT_learning import TT_spectral_learning

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils import data

def khatri_rao_torch(X, Y):
  result = [torch.ger(X[i], Y[i]) for i in range(len(X))]
  return torch.stack(result).reshape(len(X), -1)

def kronecker(x, Y):
  #print(x.shape, Y[0].shape)
  x = x.reshape(-1,)
  result = [torch.ger(x, Y[i]) for i in range(len(Y))]
  return torch.stack(result).reshape(len(Y), -1)

"""
Define the NN architecture
"""

class second_order_RNN(nn.Module):
    def __init__(self, rank, input_dim, output_dim, length, alpha = None, omega = None, transition = None):
        super(second_order_RNN, self).__init__()
        if alpha is None:
            self.transition_alpha = torch.nn.Parameter(torch.nn.init.uniform_(torch.FloatTensor(input_dim, rank), -1, 1))
            self.transition_omega = torch.nn.Parameter(torch.nn.init.uniform_(torch.FloatTensor(rank, output_dim), -1, 1))
            self.transition = torch.nn.Parameter(torch.nn.init.uniform_(torch.FloatTensor(rank, input_dim, rank), -1, 1))
        else:
            self.transition_alpha = torch.nn.Parameter(torch.from_numpy(alpha), requires_grad=True)
            self.transition_omega = torch.nn.Parameter(torch.from_numpy(omega.reshape(omega.shape[0], -1)), requires_grad=True)
            self.transition = torch.nn.Parameter(torch.from_numpy(transition), requires_grad=True)

        self.length = length
        self.rank = rank
        self.input_dim = input_dim
        self.output_dim = output_dim
        #self.myparameters = nn.ParameterList(self.transition_alpha, self.transition_omega, self.transition)

    def forward(self, x):
        #print(x)
        #assert x.shape[1] == self.input_dim, 'input dimension mismatches network structure'
        #assert x.shape[2] == self.length, 'input length mismatches network structure'
        #temp =
        #print(temp.shape)
        for i in range(x.shape[2]):
            if i ==0:
                #print(x.shape, self.transition_alpha.shape)
                #temp = torch.matmul(self.transition_alpha, x[:, :, 0])
                temp = self.kronecker(self.transition_alpha, x[:, :, 0])
                temp = torch.mm(temp, self.transition.reshape(self.rank * self.input_dim, self.rank))
                continue
            #print(temp.shape)
            temp = self.khatri_rao_torch(temp, x[:, :, i])
            #print(temp.shape)
            temp = torch.mm(temp, self.transition.reshape(self.rank*self.input_dim, self.rank))
        #print(temp.shape, self.transition_omega.shape)
        temp = torch.mm(temp, self.transition_omega)
        return temp

    def khatri_rao_torch(self, X, Y):
        result = [torch.ger(X[i], Y[i]) for i in range(len(X))]
        return torch.stack(result).reshape(len(X), -1)

    def kronecker(self, x, Y):
        # print(x.shape, Y[0].shape)
        x = x.reshape(-1, )
        result = [torch.ger(x, Y[i]) for i in range(len(Y))]
        return torch.stack(result).reshape(len(Y), -1)

def tic():
    return time.clock()

def toc(t):
    return time.clock() - t

def cal_RMSE(pred, ytest, mean_data, std_data):
    pred = (pred * std_data) + mean_data
    ytest = (ytest * std_data) + mean_data
    return np.sqrt(np.mean((pred - ytest) ** 2))


def cal_MAPE(pred, ytest, mean_data, std_data):
    pred = (pred * std_data) + mean_data
    ytest = (ytest * std_data) + mean_data
    return np.mean(np.abs(pred - ytest) / ytest)


def cal_MAE(pred, ytest, mean_data, std_data):
    pred = (pred * std_data) + mean_data
    ytest = (ytest * std_data) + mean_data
    return np.mean(np.abs(pred - ytest))

def recover_tensor(tt):
    temp = tt[0]
    for i in range(1, len(tt)):
        #print(temp.shape, tt[i].shape)
        temp = np.tensordot(temp, tt[i], axes=[[-1], [0]])
    return temp
#def padding(X):
class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, x, y):
        'Initialization'
        self.x = x
        self.y = y

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x, y
criterion = nn.MSELoss()
lr = 0.01

def fine_tune(train_X, train_Y, test_X, test_Y, alpha, omega, A,
              criterion = nn.MSELoss(), lr = 0.001, n_epochs =100):

    input_dim = train_X.shape[1]
    train_Y = train_Y.reshape(train_Y.shape[0], -1)
    test_Y = test_Y.reshape(test_Y.shape[0], -1)
    output_dim = train_Y.shape[1]
    length = train_X.shape[2]
    rank = A.shape[0]
    model = second_order_RNN(rank, input_dim, output_dim, length, alpha, omega, A)
    training_set = Dataset(train_X, train_Y)
    vali_set = Dataset(test_X, test_Y)
    num_workers = 0
    batch_size = 256
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size,
                                               num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(vali_set, batch_size=batch_size,
                                              num_workers=num_workers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    model.train()  # prep model for training

    training_loss = []
    testing_loss = []
    for epoch in range(n_epochs):
        # monitor training loss
        train_loss = 0.0
        test_loss = 0.0
        ###################
        # train the model #
        ###################
        for data, target in train_loader:
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            # data = data.to(device)
            # target = target.to(device)
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item() * data.size(0)

        for data, target in test_loader:
            # forward pass: compute predicted outputs by passing inputs to the model
            # data = data.to(device)
            # target = target.to(device)

            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # update test loss
            test_loss += loss.item() * data.size(0)

        # print training statistics
        # calculate average loss over an epoch
        train_loss = train_loss / len(train_loader.dataset)
        test_loss = test_loss / len(test_loader.dataset)
        training_loss.append(train_loss)
        testing_loss.append(test_loss)
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch + 1, train_loss, test_loss))
        #print(time.clock() - t)
    return model

def convert_pytorch_to_linRNN(model):
    from LinRNN import LinRNN
    return LinRNN(model.transition_alpha.detach().numpy(), model.transition.detach().numpy(), model.transition_omega.detach().numpy())


if __name__ == '__main__':
    '''
    python Run_Experiment.py -exp Addition -ns 2 -nr 1 -var 0.1 -a 1. -aepo 10
    nohup python -u Run_Experiment.py -exp Pollution -ns 5 -nr 1 -var 0.1 -a 1. -aepo 10 -lne 10000 -lm TIHT > pollution5.out
    '''

    #L_num_examples = [20, 40, 80, 160, 320, 640, 1500, 2560, 5000]
    L_num_examples = [1000]
    target_file_name = 'target_working.pickle'
    target_num_states = 5
    target_input_dim = 3
    target_output_dim = 2
    N_runs = 1
    length = 2
    test_length = 6
    methods = ['ALS']
    TIHT_epsilon = 1e-15
    TIHT_learning_rate = 1e-1
    TIHT_max_iters = 10000
    xp_path = './Default_experiment_folder/'
    exp = 'Addition'
    b2 = 100
    lr2 = 0.001
    epo2 = 1000
    tol = 50
    verbose = False

    ALS_epochs = 3

    '''Parser set up'''
    parser = argparse.ArgumentParser()

    '''General experiment specification'''
    parser.add_argument('-exp', '--experiment_name', help = 'name of experiments, Addition, RandomRNN or Wind')
    parser.add_argument('-lne', '--list_number_examples', nargs = '+', help='list of examples numbers', type=int)
    parser.add_argument('-nr', '--number_runs', help='number of runs', type=int)
    parser.add_argument('-le', '--length', help='minimum training length', type=int)
    parser.add_argument('-tle', '--testing_length', help='testing length', type=int)
    parser.add_argument('-xp', '--xp_path', help='experiment folder path')
    parser.add_argument('-lm', '--method_list', nargs='+', help="List of methods to use, can be IHT, TIHT, NuclearNorm, "
                                                                "OLS, ALS, LSTM, SGD+TIHT, TIHT_lowmem")

    '''If using TIHT/IHT specify the following'''
    parser.add_argument('-eps', '--HT_epsilon', help='epsilon for TIHT and IHT', type=float)
    parser.add_argument('-lr', '--HT_learning_rate', help='learning rate for TIHT and IHT', type=float)
    parser.add_argument('-mi', '--HT_max_iter', help='number of max iterations for TIHT and IHT', type=int)

    '''If using NuclearNorm method, specify the following'''
    parser.add_argument('-a', '--alpha', help='hyperparameter for nuclear norm method', type=float)

    '''If using ALS, specify the following'''
    parser.add_argument('-aepo', '--ALS_epoches', help='Number of epochs when using ALS', type=int)

    '''If running Random2RNN exp, and launching a new experiment, specify the following'''
    parser.add_argument('-var', '--noise', help='variance of the gaussian noise', type=float)
    parser.add_argument('-ns', '--states_number', help='number of states for the model', type=int)
    parser.add_argument('-ld', '--load_data', default=False, help='load the previously created data', action='store_true')
    parser.add_argument('-tfn', '--target_file_name', help='target file name')
    parser.add_argument('-tns', '--target_number_states', help='number of states for the target 2-rnn', type=int)
    parser.add_argument('-tid', '--target_input_dimension', help='input dimension for the target 2-rnn', type=int)
    parser.add_argument('-tod', '--target_output_dimension', help='output dimension for the target 2-rnn', type=int)
    parser.add_argument('-lt', '--load_target', default=False, help='load the previously created target 2rnn', action='store_true')

    '''If running TIHT+SGD, specify the following'''
    parser.add_argument('-lr2', help='learning rate for sgd 2rnn', type=float)
    parser.add_argument('-epo2', help='number of epochs for sgd 2rnn', type=int)
    parser.add_argument('-b2', '--batch_size', help='batch size for sgd 2rnn', type=int)
    parser.add_argument('-t', '--tolerance', help='tolerance for sgd 2rnn', type=int)
    args = parser.parse_args()

    '''Arguments set up'''
    if args.experiment_name != None:
        exp = args.experiment_name
    else:
        raise Exception('Did not initialize which experiment to run, try set up after -exp argument')
    if args.noise != None:
        noise_level = args.noise
    else:
        raise Exception('Did not initialize noise_level, try set up after -var argument')
    if args.states_number != None:
        num_states = args.states_number
    else:
        raise Exception('Did not initialize state numbers, try set up after -ns argument')
    if args.alpha != None:
        alpha = args.alpha
    else:
        raise Exception('Did not initialize alpha, try set up after -a argument')

    if args.load_data:
        load_data = True
    else:
        load_data = False

    if args.load_target:
        load_target = True
    else:
        load_target = False
    if args.number_runs:
        num_runs = args.number_runs
    if args.list_number_examples:
        L_num_examples = args.list_number_examples
    if args.number_runs:
        N_runs = args.number_runs
    if args.length:
        length = args.length
    if args.ALS_epoches:
        ALS_epochs = args.ALS_epoches
    if args.testing_length:
        test_length = args.testing_length
    if args.method_list:
        methods = args.method_list
    if args.HT_epsilon:
        TIHT_epsilon = args.HT_epsilon
    if args.HT_learning_rate:
        TIHT_learning_rate = args.HT_learning_rate
    if args.HT_max_iter:
        TIHT_max_iters = args.HT_max_iter
    if args.xp_path:
        xp_path = args.xp_path

    if args.lr2:
        lr2 = args.lr2
    if args.epo2:
        epo2 = args.epo2
    if args.batch_size:
        b2 = args.batch_size
    if args.tolerance:
        tol = args.tolerance

    if args.target_file_name:
        target_file_name = args.target_file_name
    if args.target_number_states:
        target_num_states = args.target_number_states
    if args.target_input_dimension:
        target_input_dim = args.target_input_dimension
    if args.target_output_dimension:
        target_output_dim = args.target_output_dimension

    '''Folder set up and results savers set up'''
    if exp =='Addition':
        xp_path = './Data/Addition'
    elif exp == 'RandomRNN':
        xp_path = './Data/RandomRNN'
    if not os.path.exists(xp_path):
        os.makedirs(xp_path)
    if not os.path.exists(xp_path + 'noise_' + str(noise_level)):
        os.makedirs(xp_path + 'noise_' + str(noise_level))
    xp_path = xp_path + 'noise_' + str(noise_level)+'/'

    if not os.path.exists(xp_path):
        os.makedirs(xp_path)

    results = dict([(m, {}) for m in methods])

    for num_examples in L_num_examples:
        for m in methods:
            results[m][num_examples] = []

    results['NUM_EXAMPLES'] = L_num_examples

    times = dict([(m, {}) for m in methods])

    for num_examples in L_num_examples:
        for m in methods:
            times[m][num_examples] = []

    times['NUM_EXAMPLES'] = L_num_examples

    '''Generate corresponding experiment data'''
    if exp == 'Addition':
        print(load_data)
        if load_data == False:
            data_function = lambda l, n: synthetic_data.generate_data_simple_addition(num_examples = n, seq_length = l, noise_level=noise_level)
            Xtest, ytest = data_function(n = 1000, l = test_length)
            with open(xp_path + '/Test.pickle', 'wb') as f:
                pickle.dump([Xtest, ytest], f)

        else:
            data_function = lambda l, n: synthetic_data.generate_data_simple_addition(num_examples = n, seq_length = l, noise_level=noise_level)
            with open('./Data/Addition/noise_' + str(noise_level) + '/Test.pickle', 'rb') as f:
                [Xtest, ytest] = pickle.load(f)

        print("test MSE of zero function", np.mean(ytest ** 2))

    elif exp == 'RandomRNN':
        if load_data == False:
            if load_target == True:
                with open(target_file_name, 'rb') as f:
                    target = pickle.load(f)
            else:
                target = synthetic_data.generate_random_LinRNN(target_num_states, target_input_dim, target_output_dim,
                                                               alpha_variance=0.2, A_variance=0.2,
                                                               Omega_variance=0.2)
                with open(target_file_name, 'wb') as f:
                    pickle.dump(target, f)

            data_function = lambda l, n: synthetic_data.generate_data(target, N_samples = n, seq_length = l,
                                                                   noise_variance=noise_level)
            Xtest, ytest = data_function(n = 1000, l = test_length)
            with open(xp_path + 'all_data.pickle', 'wb') as f:
                pickle.dump([Xtest, ytest], f)
        else:
            with open('./Data/RandomRNN/noise_' + str(noise_level) + '_units_' + str(target_num_states) + '/Test.pickle',
                    'rb') as f:
                [Xtest, ytest] = pickle.load(f)
            with open(target_file_name, 'rb') as f:
                target = pickle.load(f)
            data_function = lambda l, n: synthetic_data.generate_data(target, n, l, noise_variance=noise_level)
    elif exp == 'Wind':
        data, train_test_split = synthetic_data.generate_wind_speed('./Data/Wind_Speed/train.csv', './Data/Wind_Speed/test.csv')
        mean_data = np.  mean(data)
        std_data = np.std(data)
        data = (data - mean_data) / std_data
        data_function_train = lambda l: synthetic_data.generate_wind_train(data, train_test_split, l)
        data_function_test = lambda l: synthetic_data.generate_wind_test(data, train_test_split, l)
    elif exp == 'Pollution':
        data, scaler = pollution.prepare_data()
        training_data = data[:-1000]
        test_data = data[-1000:]
        data_function = lambda l, n: pollution.generate_pollution_data(training_data, l, n)
        data_function_test = lambda l: pollution.generate_pollution_data(test_data, l)
        test_length = 2*length
        Xtest, ytest = data_function_test(l=test_length)

    else:
        raise Exception('Experiment not found')
    '''Run experiment'''

    for run in range(num_runs):
        for num_examples in L_num_examples:
            print('______\nsample size:', num_examples)
            print('Current Experiment: ' + str(exp) +' with noise ' + str(noise_level) + ' and ' + str(num_states) + ' states')
            #data_function = lambda l: generate_data_simple_addition(num_examples, l, noise_level=noise_level)
            Xl, yl = data_function(n = num_examples, l = length)
            X2l, y2l = data_function(n = num_examples, l = length * 2)
            X2l1, y2l1 = data_function(n = num_examples, l = length * 2 + 1)
            # data_function = lambda l: generate_data_simple_addition2(1000, l, noise=noise_level)
            #Xtest, ytest = data_function(1000, test_length)

            for method in methods:
                if method == 'TIHT' and exp == 'Pollution':
                    X = []
                    Y = []
                    for i in range(length * 2 + 2):
                        tempx, tempy = data_function(n = num_examples, l = i)


                        X.append(tempx)
                        Y.append(tempy)
                    H_vec = []
                    temp_index_list = [length, 2 * length, 2 * length + 1]

                    for i in temp_index_list:
                        X_temp_vec = []
                        Y_temp_vec = []
                        # print(i)
                        for j in range(1, i + 1):
                            X_temp = X[j]
                            Y_temp = Y[j].reshape(-1, 1)

                            X_temp_vec.append(X_temp)
                            Y_temp_vec.append(Y_temp)

                            X_temp = np.swapaxes(X_temp, 1, 2)
                            # print(X_temp[0:5])
                            # print(Y_temp[0:5])
                            # Tl = learning.sequence_to_tensor(X_temp)
                            #
                            # H_temp = learning.approximate_hankel(Tl, Y_temp, alpha_ini_value=1.,
                            #                                      rank=num_states, eps=TIHT_epsilon,
                            #                                      learning_rate=TIHT_learning_rate,
                            #                                      max_iters=TIHT_max_iters,
                            #                                      method='TIHT', verbose=-1)
                            # print(H_temp, H_temp.shape)
                        X_temp, Y_temp = synthetic_data.pad_data(X_temp_vec, Y_temp_vec)
                        X_temp = np.swapaxes(X_temp, 1, 2)
                        Tl = learning.sequence_to_tensor(X_temp)

                        H_temp = learning.approximate_hankel(Tl, Y_temp, alpha_ini_value=1.,
                                                         rank=num_states, eps=TIHT_epsilon,
                                                         learning_rate=TIHT_learning_rate, max_iters=TIHT_max_iters,
                                                         method='TIHT', verbose=-1)
                        #print(H_temp)
                        H_temp = H_temp.squeeze()
                        #H_temp = recover_tensor(H_temp).squeeze()
                        H_temp_shape = list(H_temp.shape)
                        H_temp_shape.append(1)
                        H_temp = H_temp.reshape(tuple(H_temp_shape))
                        print(H_temp.shape)
                        H_vec.append(H_temp)
                        # print(H_temp)
                    learned_model = learning.spectral_learning(num_states, H_vec[1], H_vec[2], H_vec[0])

                    #T = toc(t)
                    #Xtest = np.swapaxes(Xtest, 1, 2)
                    pred = []
                    x_test_temp, y_test_temp = synthetic_data.pad_data([Xtest], [ytest.reshape(-1, 1)])
                    #x_test_temp = np.swapaxes(x_test_temp, 1, 2)
                    print(x_test_temp.shape)
                    x_test_temp = np.swapaxes(x_test_temp, 1, 2)
                    for o in x_test_temp:
                        pred.append(learned_model.predict(o))
                    pred = np.array(pred).reshape(len(ytest), )
                    ytest = np.array(ytest).reshape(len(ytest), )
                    test_mse = np.mean((pred - ytest) ** 2)
                    print(method, "test MSE:", test_mse)
                    #ytest = (ytest - scaler.min_[4]) / scaler.scale_[4]
                    #pred = (pred - scaler.min_[4]) / scaler.scale_[4]
                    ytest = ytest * (scaler.var_[4]**0.5) + scaler.mean_[4]
                    pred = pred * (scaler.var_[4]**0.5) + scaler.mean_[4]
                    #print(scaler.min_[4], scaler.scale_[4])
                    print(pred[0:5])
                    print(ytest[0:5])
                    test_mse = np.mean((pred - ytest) ** 2)
                    print(method, "test MSE:", test_mse)

                # print(method)
                elif method != 'LSTM' and method != 'TIHT+SGD' and method != 'ALS':
                    Xl_here = np.swapaxes(Xl, 1, 2)
                    X2l_here = np.swapaxes(X2l, 1, 2)
                    X2l1_here = np.swapaxes(X2l1, 1, 2)
                    Xtest_here = np.swapaxes(Xtest, 1, 2)
                    Tl = learning.sequence_to_tensor(Xl_here)
                    T2l = learning.sequence_to_tensor(X2l_here)
                    T2l1 = learning.sequence_to_tensor(X2l1_here)
                    t = tic()
                    Hl = learning.approximate_hankel(Tl, yl, alpha_ini_value=alpha,
                                                     rank=num_states, eps=TIHT_epsilon,
                                                     learning_rate=TIHT_learning_rate, max_iters=TIHT_max_iters,
                                                     method=method, verbose=-1)
                    H2l = learning.approximate_hankel(T2l, y2l, alpha_ini_value=alpha,
                                                      rank=num_states, eps=TIHT_epsilon,
                                                      learning_rate=TIHT_learning_rate, max_iters=TIHT_max_iters,
                                                      method=method, verbose=-1)
                    H2l1 = learning.approximate_hankel(T2l1, y2l1, alpha_ini_value=alpha, rank=num_states, eps=TIHT_epsilon,
                                                       learning_rate=TIHT_learning_rate, max_iters=TIHT_max_iters,
                                                       method=method, verbose=-1)
                    print('Hankel building time', toc(t))
                    t = tic()
                    if method == 'TIHT_lowmem':
                        import TT_learning
                        learned_model = TT_learning.TT_spectral_learning(H2l, H2l1, Hl)
                    else:
                        learned_model = learning.spectral_learning(num_states, H2l, H2l1, Hl)
                        #print(H2l.shape, H2l1.shape, Hl.shape)

                    test_mse = learning.compute_mse(learned_model, Xtest_here, ytest)
                    train_mse = learning.compute_mse(learned_model, X2l1_here, y2l1)
                    # print(test_mse)
                    if train_mse > np.mean(y2l1 ** 2):
                        test_mse = np.mean(ytest ** 2)
                    print(method, "test MSE:", test_mse, "\t\t spectral leanring time:", toc(t))

                    results[method][num_examples].append(test_mse)
                    times[method][num_examples].append(toc(t))
                elif method == 'LSTM':

                    def padding_function(x, desired_length):
                        if desired_length <= x.shape[1]:
                            return x
                        x = np.insert(x, x.shape[1], np.zeros((desired_length - x.shape[1], 1, x.shape[2])), axis=1)
                        return x
                    Xl_padded = padding_function(Xl, test_length)
                    X2l_padded = padding_function(X2l, test_length)
                    X2l1_padded = padding_function(X2l1, test_length)
                    X = np.concatenate((Xl_padded, X2l_padded, X2l1_padded))
                    Y = np.concatenate((yl, y2l, y2l1))
                    t = tic()
                    learned_model = learning.RNN_LSTM(X, Y, test_length, num_states, noise_level, 'RandomRNN')
                    test_mse = learning.compute_mse(learned_model, Xtest, ytest, lstm=True)
                    train_mse = learning.compute_mse(learned_model, X2l1_padded, y2l1, lstm=True)
                    # if train_mse > np.mean(y2l1 ** 2):
                    #   test_mse = np.mean(ytest ** 2)
                    print(method, "test MSE:", test_mse, "\t\ttime:", toc(t))
                    results[method][num_examples].append(test_mse)
                    times[method][num_examples].append(toc(t))
                elif method == 'TIHT+SGD':
                    X = []
                    Y = []
                    for i in range(length * 2 + 2):
                        tempx, tempy = data_function(i, num_examples)
                        X.append(tempx)
                        Y.append(tempy)
                    t = tic()
                    if noise_level == 0.:
                        TIHT_learning_rate = 0.000001
                    learned_model = learning.TIHT_SGD_torch(X, Y, num_states, length, verbose, TIHT_epsilon,
                                                            TIHT_learning_rate,
                                                            TIHT_max_iters,
                                                            lr2, epo2, b2, tol, alpha=1., lifting=False)

                    test_mse = learning.compute_mse(learned_model, Xtest, ytest, if_tc=True)
                    train_mse = learning.compute_mse(learned_model, X2l1, y2l1, if_tc=True)
                    if train_mse > np.mean(y2l1 ** 2):
                        test_mse = np.mean(ytest ** 2)
                    print(method, "test MSE:", test_mse, "\t\ttime:", toc(t))
                    results[method][num_examples].append(test_mse)
                    times[method][num_examples].append(toc(t))

                elif method == 'ALS':
                    #print(Xl.shape, X2l.shape, X2l1.shape, Xtest.shape)
                    if exp == 'Pollution':
                        X = []
                        Y = []
                        for i in range(length * 2 + 2):
                            tempx, tempy = data_function(i, num_examples)

                            #tempx = np.delete(tempx, 0, axis  = 2)
                            print(tempy.shape)
                            X.append(tempx)
                            Y.append(tempy)
                        H_vec = []
                        temp_index_list = [length, 2 * length, 2 * length + 1]
                        for i in temp_index_list:
                            X_temp_vec = []
                            Y_temp_vec = []
                            # print(i)
                            for j in range(1, i + 1):
                                # print('j', j)
                                #X_temp = np.swapaxes(X[j], 1, 2)
                                X_temp = X[j]
                                Y_temp = Y[j].reshape(-1, 1)
                                X_temp_vec.append(X_temp)
                                Y_temp_vec.append(Y_temp)
                            X_temp, Y_temp = synthetic_data.pad_data(X_temp_vec, Y_temp_vec)
                            H_temp = learning.ALS(X_temp, Y_temp, rank=num_states, X_vali=None, Y_vali=None,
                                                  n_epochs=ALS_epochs)
                            H_temp = recover_tensor(H_temp).squeeze()
                            H_temp_shape = list(H_temp.shape)
                            H_temp_shape.append(1)
                            H_temp = H_temp.reshape(tuple(H_temp_shape))
                            H_vec.append(H_temp)
                        learned_model = learning.spectral_learning(num_states, H_vec[1], H_vec[2], H_vec[0])
                        #Xtest = np.delete(Xtest, 0, axis=2)
                        Xtest, ytest = synthetic_data.pad_data([Xtest], [ytest.reshape(-1, 1)])

                        print(X_temp.shape, Y_temp.shape, Xtest.shape)
                        # model = fine_tune(X_temp, Y_temp, Xtest, ytest, 1./(num_states**0.5)*np.ones((num_states, )),
                        #                   1./(num_states**0.5)*np.ones((num_states, 1)),
                        #                   1./((num_states*X_temp.shape[1]*num_states)**0.5)*np.ones((num_states, X_temp.shape[1], num_states)),
                        #           criterion=nn.MSELoss(), lr=0.001, n_epochs=100)
                        # learned_model = convert_pytorch_to_linRNN(model)
                    else:
                        t = tic()
                        H_l_cores = learning.ALS(Xl, yl, rank=num_states, X_vali=None, Y_vali=None, n_epochs=ALS_epochs)
                        H_2l_cores = learning.ALS(X2l, y2l, rank=num_states, X_vali=None, Y_vali=None, n_epochs=ALS_epochs)
                        H_2l1_cores = learning.ALS(X2l1, y2l1, rank=num_states, X_vali=None, Y_vali=None, n_epochs=ALS_epochs)
                        print('Hankel building time:', toc(t))
                        H_l = recover_tensor(H_l_cores).squeeze()
                        H_2l = recover_tensor(H_2l_cores).squeeze()
                        H_2l1 = recover_tensor(H_2l1_cores).squeeze()
                        print('Hankel building time:', toc(t))
                        t = tic()
                        Hl_shape = list(H_l.shape)
                        H2l_shape = list(H_2l.shape)
                        H2l1_shape = list(H_2l1.shape)
                        if ytest.shape[1] == 1:
                            Hl_shape.append(1)
                            H2l1_shape.append(1)
                            H2l_shape.append(1)
                            H_l = H_l.reshape(tuple(Hl_shape))
                            H_2l = H_2l.reshape(tuple(H2l_shape))
                            H_2l1 = H_2l1.reshape(tuple(H2l1_shape))
                        learned_model = learning.spectral_learning(num_states, H_2l, H_2l1, H_l)
                        #learned_model = TT_spectral_learning(H_2l, H_2l1, H_l)
                        print('Spectral learning time:', toc(t))
                    #learned_model.alpha = learned_model.alpha.reshape(-1, 1)
                    #print(learned_model.alpha, learned_model.A.shape, learned_model.Omega.shape)
                    #if exp == 'Addition':
                    Xtest_temp = np.swapaxes(Xtest, 1, 2)
                    train = np.swapaxes(X2l1, 1, 2)
                    #else:
                    #Xtest_temp = Xtest
                    #train = X2l1
                    #print(Xtest_temp.shape)
                    if exp == 'Pollution':
                        #test_mse = learning.compute_mse(learned_model, Xtest_temp, ytest)
                        pred = []
                        for o in Xtest_temp:
                            pred.append(learned_model.predict(o))
                        pred = np.array(pred).reshape(len(ytest), -1)
                        # ytest = (ytest - scaler.min_[4]) / scaler.scale_[4]
                        # pred = (pred - scaler.min_[4]) / scaler.scale_[4]
                        # print(scaler.min_[4], scaler.scale_[4])
                        ytest = ytest * (scaler.var_[4] ** 0.5) + scaler.mean_[4]
                        pred = pred * (scaler.var_[4] ** 0.5) + scaler.mean_[4]
                        # print(pred)
                        # print(ytest)
                        # print(pred.shape, ytest.shape)
                        pred = pred.reshape(-1, )
                        ytest = ytest.reshape(-1, )
                        test_mse = np.mean((pred - ytest) ** 2)
                    else:
                        test_mse = learning.compute_mse(learned_model, Xtest_temp, ytest)
                        train_mse = learning.compute_mse(learned_model, train, y2l1)
                        if train_mse > np.mean(y2l1 ** 2):
                            test_mse = np.mean(ytest ** 2)
                    print(method, "test MSE:", test_mse)
                    results[method][num_examples].append(test_mse)

            with open(xp_path + 'results_' + str(num_states) + '_states.pickle', 'wb') as f:
                pickle.dump(results, f)
    print(results)