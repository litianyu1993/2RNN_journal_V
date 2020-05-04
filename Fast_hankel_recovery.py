import numpy as np
import tensornetwork as tn
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils import data
import time
import tensorly as tl
def tic():
    return time.clock()

def toc(t):
    return time.clock() - t
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
    def __init__(self, rank, input_dim, output_dim, length):
      super(second_order_RNN, self).__init__()
      self.transition_alpha = torch.nn.Parameter(torch.nn.init.uniform_(torch.FloatTensor(input_dim, rank), -1, 1))
      self.transition_omega = torch.nn.Parameter(torch.nn.init.uniform_(torch.FloatTensor(rank, output_dim), -1, 1))
      self.transition = torch.nn.Parameter(torch.nn.init.uniform_(torch.FloatTensor(rank, input_dim, rank), -1, 1))
      self.length = length
      self.rank = rank
      self.input_dim = input_dim
      self.output_dim = output_dim

    def forward(self, x):
      #print(x)
      assert x.shape[1] == self.input_dim, 'input dimension mismatches network structure'
      assert x.shape[2] == self.length, 'input length mismatches network structure'
      #temp =
      #print(temp.shape)
      for i in range(self.length):
        if i ==0:
          temp = torch.matmul(x[:, :, 0], self.transition_alpha)
          continue
        temp = khatri_rao_torch(temp, x[:, :, i])
        temp = torch.mm(temp, self.transition.reshape(self.rank*self.input_dim, self.rank))
      temp = torch.mm(temp, self.transition_omega)
      return temp

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

## Define the NN architecture
class Net(nn.Module):
    def __init__(self, rank, input_dim, output_dim, length):
      super(Net, self).__init__()
      self.layers = [torch.nn.Parameter(torch.nn.init.uniform_(torch.FloatTensor(input_dim, rank), -1.0/np.sqrt(input_dim), 1.0/np.sqrt(input_dim) ))]
      for i in range(length-1):
        self.layers.append(torch.nn.Parameter(torch.nn.init.uniform_(torch.FloatTensor(rank, input_dim, rank), -1.0/np.sqrt(input_dim), 1.0/np.sqrt(input_dim))))
      self.layers.append(torch.nn.Parameter(torch.nn.init.uniform_(torch.FloatTensor(rank, output_dim), -1.0/np.sqrt(rank), 1.0/np.sqrt(rank))))
      self.layers = nn.ParameterList( self.layers )
      self.length = length
      self.rank = rank
      self.input_dim = input_dim
      self.output_dim = output_dim

      self.lnorm1 = nn.LayerNorm([input_dim,length])

    def forward(self, x):
      #print(x)
      assert x.shape[1] == self.input_dim, 'input dimension mismatches network structure'
      assert x.shape[2] == self.length, 'input length mismatches network structure'
      x = self.lnorm1( x )
      for i in range(self.length):
        if i ==0:
          temp = torch.matmul(x[:, :, 0], self.layers[i])
          continue
        temp = khatri_rao_torch(temp, x[:, :, i])
        temp = torch.mm(temp, self.layers[i].reshape(self.rank*self.input_dim, self.rank))
      temp = torch.mm(temp, self.layers[-1])
      return temp


def train_gradient_descent_standard(X, Y, X_val, Y_val, model, optimizer, criterion, n_epochs, device, batch_size=256):
    training_set = Dataset(X, Y)
    vali_set = Dataset(X_val, Y_val)
    # number of subprocesses to use for data loading
    num_workers = 0
    # how many samples per batch to load
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size,
                                               num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(vali_set, batch_size=batch_size,
                                              num_workers=num_workers)

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
            data = data.to(device)
            target = target.to(device)
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
            data = data.to(device)
            target = target.to(device)

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
    return training_loss, testing_loss, model


def khatri_rao(X, Y):
    #result = [np.outer(X[i], Y[i]) for i in range(len(X))]
    result = tl.tenalg.khatri_rao([X.T, Y.T]).T
    return np.asarray(result).reshape(len(X), -1)


def tensor_X_smaller_than_index(X, index):
    assert index > 0, 'index should be larger than zero'
    X_ten = X[:, :, 0]
    for i in range(1, index):
        X_ten = khatri_rao(X_ten, X[:, :, i])
    return X_ten


def tensor_X_larger_than_index(X, index):
    assert index < X.shape[2] - 1, 'index should be smaller than the maximum length'
    assert index > 0, 'index should be larger than zero'
    X_ten = X[:, :, 0]
    for i in range(index, X.shape[2]):
        X_ten = khatri_rao(X_ten, X[:, :, i])
    return X_ten



# Create the nodes
def create_TT(X, Y, rank):
    max_length = X.shape[2]
    dimension = X.shape[1]
    out_dim = Y.shape[1]
    # return a list of tensor_train nodes
    #print(dimension, rank)
    a = (np.random.rand(dimension, rank))
    #tensor_train = [tn.Node(2 * (np.random.rand(dimension, rank) - 0.5))]
    tensor_train = [tn.Node(np.random.normal(0, (1./np.sqrt(len(X))), (dimension, rank)))]
    for i in range(1, max_length - 1):
        #tensor_train.append(tn.Node(2 * (np.random.rand(rank, dimension, rank) - 0.5)))
        tensor_train.append(tn.Node(np.random.normal(0, (1./np.sqrt(len(X))), (rank, dimension, rank))))
    # print(out_dim)
    #tensor_train.append(tn.Node(2 * (np.random.rand(rank, dimension, out_dim) - 0.5)))
    tensor_train.append(tn.Node(np.random.normal(0, (1./np.sqrt(len(X))), (rank, dimension, out_dim))))
    # print(tensor_train[-1].shape)
    return tensor_train


def traverse_tt(tensor_train, X, start_core_number=0, end_core_number=None):
    if end_core_number is None:
        end_core_number = X.shape[2]
    assert end_core_number > 0, 'need this to be bigger than 0'
    rank = tensor_train[0].shape[1]
    dim = tensor_train[0].shape[0]
    out_dim = tensor_train[-1].shape[-1]
    count = 1
    if start_core_number == 0:
        a = tensor_train[0]
        b = tn.Node(X[:, :, 0])
        # print(a.shape, b.shape)
        edge = a[0] ^ b[1]
        temp = np.asarray(tn.contract(edge).tensor).transpose()

        # print(temp.shape)
        for i in range(1, end_core_number):
            count += 1
            if i != X.shape[2] - 1:
                # print('temp', temp.shape)
                temp = khatri_rao(temp, X[:, :, i])
                b = tensor_train[i].tensor
                b = b.reshape(rank * dim, rank)
                # print(temp.shape, b.shape)
                temp = temp @ np.asarray(b)
            else:
                temp = khatri_rao(temp, X[:, :, i])
                b = tensor_train[i].tensor

                b = b.reshape(rank * dim, out_dim)
                # print(temp.shape, b.shape, tensor_train[i].tensor.shape, tensor_train[-1].tensor.shape, i)
                temp = temp @ np.asarray(b)
                # print(temp.shape)
        # print(count)
    else:
        a = tensor_train[start_core_number]
        b = tn.Node(X[:, :, start_core_number])
        # print(a.shape, b.shape)
        edge = a[1] ^ b[1]
        e1 = a[0]
        e2 = a[2]
        e3 = b[0]
        temp = tn.contract(edge)
        temp = temp.reorder_edges([e1, e3, e2]).tensor
        # print(temp.shape)
        for i in range(start_core_number + 1, end_core_number):
            count += 1
            temp_hid = []
            for j in range(rank):
                temp_mat = temp[j]
                temp_khatri = khatri_rao(temp_mat, X[:, :, i])
                temp_hid.append(temp_khatri)
            temp_hid = np.asarray(temp_hid)
            if i != X.shape[2] - 1:
                b = tensor_train[i].tensor
                b = b.reshape(rank * dim, rank)
                temp = temp_hid @ np.asarray(b)
            else:
                b = tensor_train[i].tensor
                b = b.reshape(rank * dim, out_dim)
                temp = temp_hid @ np.asarray(b)
        # print(count)
    return temp


def solve_cores_ALS(X, Y, rank=5, tensor_train=None):
    #Y = Y.reshape(num_examples, -1)
    out_dim = Y.shape[1]
    # if out_dim == 0:
    #     out_dim = 1
    #     print(Y)
    #     Y = Y.reshape(len(Y), 1)
    max_length = X.shape[2]
    dimension = X.shape[1]
    if tensor_train is None:
        tensor_train = create_TT(X, Y, rank)
        print(evaluate(X, Y, tensor_train))
    num_cores = len(tensor_train)
    #t = tic()
    S = traverse_tt(tensor_train, X, start_core_number=1)
    #print('1', toc(t))
    #t = tic()
    W = []
    for j in range(out_dim):
        temp_S = S[:, :, j]
        temp_khatri = khatri_rao(X[:, :, 0], np.asarray(temp_S).transpose())
        W.append(temp_khatri)
    #print('2', toc(t))
    W = np.asarray(W)
    W = tn.Node(W)
    W = W.reorder_edges([W[1], W[0], W[2]]).tensor
    W = W.reshape(-1, W.shape[2])
    Y_old = Y
    Y = Y.reshape(-1)
    # print(np.linalg.pinv(W).shape, Y.shape)
    tensor_train[0] = tn.Node((np.linalg.pinv(W) @ Y).reshape(dimension, rank))
    # print('here', tensor_train[-1].tensor.shape)
    # print(tensor_train[0])
    for i in range(1, num_cores - 1):
        #t = tic()
        P = traverse_tt(tensor_train, X, start_core_number=0, end_core_number=i)
        S = traverse_tt(tensor_train, X, start_core_number=i + 1)
        #print('3', toc(t))
        # print('here', tensor_train[-1].tensor.shape)
        W = []
        #t = tic()
        for j in range(out_dim):
            temp_S = S[:, :, j]
            temp_khatri = khatri_rao(X[:, :, i], np.asarray(temp_S).transpose())
            W.append(temp_khatri)
        #print('4', toc(t))
        W = np.asarray(W)
        W = tn.Node(W)
        W = W.reorder_edges([W[1], W[2], W[0]]).tensor

        # print(W.shape)
        new_W = []
        for j in range(out_dim):
            temp_W = W[:, :, j]
            # print(P.shape, temp_W.shape)
            temp_khatri = khatri_rao(P, temp_W)
            new_W.append(temp_khatri)
        W = tn.Node(np.asarray(new_W))
        W = W.reorder_edges([W[1], W[0], W[2]]).tensor
        W = W.reshape(-1, W.shape[2])
        Y = Y.reshape(-1)
        #t = tic()
        temp_W = (np.linalg.pinv(W) @ Y).reshape(rank, dimension, rank)
        #print('5', toc(t))
        # #####Orthogonalize the core #######
        # temp_W = temp_W.reshape(rank * dimension, rank)
        # svd_truncate = rank
        # U, D, V = np.linalg.svd(temp_W)
        # # print(U.shape, D.shape, V.shape)
        # temp_W = U[:, :rank].reshape(rank, dimension, rank)
        # tensor_train[i + 1] = tn.Node((np.diag(D) @ V @ tensor_train[i + 1].tensor.reshape(rank, -1)).reshape(tensor_train[i + 1].tensor.shape))

        tensor_train[i] = tn.Node(temp_W)
        # print(np.asarray(tensor_train[i].tensor).shape)
    P = traverse_tt(tensor_train, X, start_core_number=0, end_core_number=max_length - 1)
    W = khatri_rao(P, X[:, :, -1])
    # print(W.shape, Y_old.shape)
    tensor_train[-1] = tn.Node((np.linalg.pinv(W) @ Y_old).reshape(rank, dimension, out_dim))
    scalar = 1.
    for i in range(len(tensor_train)):
        #print(np.mean(tensor_train[i].tensor))
        scalar_temp = np.max(np.abs(tensor_train[i].tensor))
        tensor_train[i].tensor /= scalar_temp
        scalar *= scalar_temp
    for i in range(len(tensor_train)):
        tensor_train[i].tensor *= np.power(scalar, 1/len(tensor_train))
        #scalar *= scalar_temp
        #print(tensor_train[i].tensor)
    #tensor_train[-1].tensor *= scalar
    return tensor_train


def evaluate(X, Y, tensor_train):
    pred = traverse_tt(tensor_train, X)
    # print(pred.shape)
    pred = pred.reshape(X.shape[0], -1)
    Y = Y.reshape(X.shape[0], -1)
    # print(pred[0:5])
    # print(Y[0:5])
    # target = np.sum(X[:, :-1, :], axis = 2).reshape(X.shape[0], -1)
    # print(X[:, 0, :].shape, target.shape)
    # print('actual target:', np.mean((target - Y)**2))
    # print('predicted:', np.mean((pred - Y)**2))
    print(pred[0:5])
    print(Y[0:5])
    return np.mean((pred - Y) ** 2)


def train_ALS(X, Y, rank, X_val=None, Y_val=None, n_epochs = 50):
    tensor_train = solve_cores_ALS(X, Y, rank = rank)
    #tensor_train = normalize_cores(tensor_train)

    error = []
    test_error = []
    #target = np.sum(X[:, :-1, :], axis = 2).reshape(X.shape[0], -1)
    for i in range(n_epochs):
        tensor_train = solve_cores_ALS(X, Y, tensor_train = tensor_train, rank = rank)
        #for j in range(len(tensor_train)):
        #    print(tensor_train[j].tensor.shape)
        #tensor_train = normalize_cores(tensor_train)
        print('training error: ' + str(evaluate(X, Y, tensor_train)))
        if X_val is not None:
            error.append(evaluate(X, Y, tensor_train))
            test_error.append(evaluate(X_val, Y_val, tensor_train))
            print('training error: '+str(error[-1])+' test error: '+str(test_error[-1]))
    error.append(evaluate(X, Y, tensor_train))
    print('training error: ' + str(error[-1]))
    return error, test_error, tensor_train


def generate_simple_addition(num_examples = 100, traj_length = 5, n_dim = 1, noise_level = 0.1):
    X = np.random.rand(num_examples, n_dim+1, traj_length)
    X[:, -1, :] = np.ones((num_examples, traj_length))
    Y = np.sum(X[:, :-1, :], axis = 2)
    Y = Y.reshape(num_examples, -1) + np.random.normal(0, noise_level, [num_examples, n_dim]).reshape(num_examples, n_dim)
    return X, Y
#
# device = 'cuda:0'
#
# n_dim = 1
# traj_length = 4
# num_examples = 1000
# noise_level = 0.1
# rank = 2
# X, Y = generate_simple_addition(num_examples = num_examples, traj_length = traj_length, n_dim = n_dim, noise_level = noise_level)
# X_val, Y_val = generate_simple_addition(num_examples = int(min(num_examples*0.5, 100)), traj_length = traj_length, n_dim = n_dim, noise_level = noise_level)
# #print(Y.shape)
# n_epochs = 50
# #print('starting')
# print(X.shape, Y.shape)
# #train_error_ALS, test_error_ALS, tensor_train = train_ALS(X, Y, rank, X_val, Y_val, n_epochs)
#
# X, Y = (torch.from_numpy(X)).float(), (torch.from_numpy(Y)).float()
# # X_val, Y_val = (torch.from_numpy(X_val)).float(), (torch.from_numpy(Y_val)).float()
# # input_dim = X.shape[1]
# # output_dim = Y.reshape(len(Y), -1).shape[1]
# # print('starting')
# # model = Net(rank, input_dim, output_dim, traj_length).to(device)
# # print('starting')
# # criterion = nn.MSELoss()
# # lr = 0.02
# # optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
# # batch_size = 256
# # n_epochs = 10000
# # train_error_GD, test_error_GD, model = train_gradient_descent_standard(X, Y, X_val, Y_val, model, optimizer, criterion, n_epochs, device, batch_size)
