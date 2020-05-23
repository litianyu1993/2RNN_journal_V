import numpy as np
import tensornetwork as tn
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils import data
import time
import tensorly as tl
from LinRNN import LinRNN
import itertools
import random
from sklearn.utils import shuffle

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
      #temp = torch.nn.softmax(temp)
      #print(temp.shape)
      #temp = torch.nn.functional.softmax(temp, dim=1)
      #print(temp.shape)
      return temp


def train_gradient_descent_standard(X, Y, model, optimizer, criterion, n_epochs, X_val = None, Y_val = None,
                                    device = None, batch_size=256, verbose = False):
    import time
    t = time.clock()
    training_set = Dataset(X, Y)

    # number of subprocesses to use for data loading
    num_workers = 0
    # how many samples per batch to load
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size,
                                               num_workers=num_workers)
    if X_val and Y_val:
        vali_set = Dataset(X_val, Y_val)
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
            if device:
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
        if X_val and Y_val:
            for data, target in test_loader:
                # forward pass: compute predicted outputs by passing inputs to the model
                if device:
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
        training_loss.append(train_loss)
        if X_val and Y_val:
            test_loss = test_loss / len(test_loader.dataset)
            testing_loss.append(test_loss)
        if verbose:
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch + 1, train_loss, test_loss))
            print(time.clock() - t)
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
    tensor_train = [tn.Node(np.random.normal(0, (1./np.sqrt(dimension*rank)), (dimension, rank)))]
    for i in range(1, max_length - 1):
        #tensor_train.append(tn.Node(2 * (np.random.rand(rank, dimension, rank) - 0.5)))
        tensor_train.append(tn.Node(np.random.normal(0, (1./np.sqrt(dimension*rank*rank)), (rank, dimension, rank))))
    # print(out_dim)
    #tensor_train.append(tn.Node(2 * (np.random.rand(rank, dimension, out_dim) - 0.5)))
    tensor_train.append(tn.Node(np.random.normal(0, (1./np.sqrt(rank*dimension*out_dim )), (rank, dimension, out_dim))))
    # print(tensor_train[-1].shape)
    return tensor_train

def convert_for_save(tt):
    to_save = []
    for i in range(len(tt)):
        to_save.append(tt[i].tensor)
    return to_save

def recover_tt_from_file(file_name):
    import pickle
    tt = pickle.load(open(file_name, 'rb'))
    tensor_train = []
    for i in range(len(tt)):
        tensor_train.append(tn.Node(tt[i]))
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

def ridge_reg(X, Y, alpha):
    from sklearn import linear_model
    ridge = linear_model.Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(X, Y)
    return ridge.coef_


def solve_cores_ALS(X, Y, rank=5, tensor_train=None, alpha = 0.1, ridge = False):
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
        import pickle
        pickle.dump(convert_for_save(tensor_train), open('init_tt_'+str(max_length),'wb'))
        #print(evaluate(X, Y, tensor_train))
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
    #print(W)
    # print(np.linalg.pinv(W).shape, Y.shape)
    #tensor_train[0] = tn.Node((np.linalg.pinv(W) @ Y).reshape(dimension, rank))
    if ridge:
        tensor_train[0] = tn.Node(ridge_reg(W, Y, alpha).reshape(dimension, rank))
    else:
        tensor_train[0] = tn.Node((np.linalg.lstsq(W, Y, rcond=None)[0]).reshape(dimension, rank))


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
        #temp_W = (np.linalg.pinv(W) @ Y).reshape(rank, dimension, rank)
        if ridge:
            temp_W = tn.Node(ridge_reg(W, Y, alpha).reshape(rank, dimension, rank))
        else:
            temp_W = np.linalg.lstsq(W, Y, rcond=None)[0].reshape(rank, dimension, rank)
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
    #print(W)
    #print(tensor_train)
    #tensor_train[-1] = tn.Node((np.linalg.pinv(W) @ Y_old).reshape(rank, dimension, out_dim))
    if ridge:
        tensor_train[-1] = tn.Node(ridge_reg(W, Y_old, alpha).reshape(rank, dimension, out_dim))
    else:
        tensor_train[-1] = tn.Node(np.linalg.lstsq(W, Y_old, rcond=None)[0].reshape(rank, dimension, out_dim))
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
    #for i in range(len(pred)):
    #    print(pred[i], Y[i])
    return np.mean((pred - Y) ** 2)


def train_ALS(X, Y, rank, X_val=None, Y_val=None, n_epochs = 50, batch_size = 1024, init_tt =None):
    #batch_size = len(X)
    tensor_train = solve_cores_ALS(X, Y, tensor_train = init_tt, rank = rank)
    #tensor_train = normalize_cores(tensor_train)

    error = []
    test_error = []
    #target = np.sum(X[:, :-1, :], axis = 2).reshape(X.shape[0], -1)
    # index_list = []-
    # for j in range(int(len(X) / batch_size) - 1):
    #     index_list.append([j * batch_size, (j + 1) * batch_size])
    #index_arr = np.asarray(index_list)

    for i in range(n_epochs):
        # current_indexes = random.sample(index_list, len(index_list))
        # for j in range(len(current_indexes)):
        #     prev = current_indexes[j][0]
        #     aft = current_indexes[j][1]
        tensor_train = solve_cores_ALS(X, Y, tensor_train = tensor_train, rank = rank)
        #for j in range(len(tensor_train)):
        #    print(tensor_train[j].tensor.shape)
        #tensor_train = normalize_cores(tensor_train)
        #print('training error: ' + str(evaluate(X, Y, tensor_train)))
        if X_val is not None:
            error.append(evaluate(X, Y, tensor_train))
            test_error.append(evaluate(X_val, Y_val, tensor_train))
            #print('training error: '+str(np.sqrt(error[-1]))+' test error: '+str(np.sqrt(test_error[-1])))
        #print(tensor_train)
    error.append(evaluate(X, Y, tensor_train))
    if X_val:
        test_error.append(evaluate(X_val, Y_val, tensor_train))
        pred = traverse_tt(tensor_train, X_val)
        pred = pred.reshape(X_val.shape[0], -1)
        Y_val = Y_val.reshape(X_val.shape[0], -1)
        print('TT', X_val[0:5], pred[0:5], Y_val[0:5])

    test_error.append(0)
    print('training error: '+str(np.sqrt(error[-1]))+' test error: '+str(np.sqrt(test_error[-1])))

    return error, test_error, tensor_train


def generate_simple_addition(num_examples = 100, traj_length = 5, n_dim = 1, noise_level = 0.1):
    X = np.random.rand(num_examples, n_dim+1, traj_length)
    X[:, -1, :] = np.ones((num_examples, traj_length))
    Y = np.sum(X[:, :-1, :], axis = 2)
    Y = Y.reshape(num_examples, -1) + np.random.normal(0, noise_level, [num_examples, n_dim]).reshape(num_examples, n_dim)
    return X, Y

def recover_tensor(tt):
    temp = tt[0]
    for i in range(1, len(tt)):
        #print(i, temp.shape, tt[i].shape)
        #print(temp)
        if len(temp.shape) ==2:
            temp = np.expand_dims(temp, axis = 0)
        #print(i, temp.shape, tt[i].shape)
        temp = np.tensordot(temp, tt[i], axes=[[-1], [0]])
    return temp

def tt_SVD(tt):
    length = len(tt)
    tt[0] = tt[0].squeeze()
    #tt[-1] = tt[0].squeeze()
    print('error', tt[0].shape)
    rank = tt[0].shape[1]
    input_dim = tt[0].shape[0]
    out_dim = tt[-1].shape[-1]
    q, r = np.linalg.qr(tt[0])
    tt[0] = q
    for i in range(1, int(length/2)):
        core_current = tt[1].reshape(rank, input_dim*rank)
        core_current = r @ core_current
        core_current = core_current.reshape(rank*input_dim, rank)
        q, r = np.linalg.qr(core_current)
        tt[1] = q.reshape(rank, input_dim, rank)
    left_r = r

    q, r = np.linalg.qr(tt[-1].reshape(rank, input_dim*out_dim))
    tt[-1] = r.reshape(rank, input_dim, out_dim)
    # for i in range(length):
    #     print(i, tt[i].shape)
    for i in range(length - 2, int(length/2), -1):
        #print(i)
        core_current = tt[i].reshape(rank*input_dim, rank)
        core_current = core_current @ q
        core_current = core_current.reshape(rank, input_dim*rank)
        q, r = np.linalg.qr(core_current)
        tt[1] = r.reshape(rank, input_dim, rank)
    right_q = q

    center = left_r @ right_q
    U, D, V = np.linalg.svd(center)
    left = U @ np.diag(D)
    right = V
    tt[int(length/2)-1] = tt[int(length/2)-1]@left
    tt[int(length/2)] = right @ tt[int(length/2)]
    return tt



def convert_to_array(tt):
    tt_arr = []
    tt_arr.append(tt[0].tensor.reshape(1, tt[0].shape[0], tt[0].shape[1]))
    for i in range(1, len(tt)):
        tt_arr.append(tt[i].tensor)
    return tt_arr
def spectral_learning(rank, H_2l, H_2l1, H_l):
    if H_2l.ndim % 2 == 0: # scalar outputs
        out_dim = 1
        l = H_l.ndim
    else:
        out_dim = H_2l.shape[-1]
        l = H_l.ndim - 1

    d = H_l.shape[0]
    #print(H_2l.shape)
    U, s, V = np.linalg.svd(H_2l.reshape([d ** l, d ** l * out_dim]))
    #print('singular:', s)
    idx = np.argsort(s)[::-1]
    #print('rank', idx, rank, U.shape, V.shape)
    U = U[:, idx[:rank]]
    V = V[idx[:rank], :]
    s = s[idx[:rank]]
    max_sin = s[0]
    #print('singular values', s)

    # s /= max_sin
    # H_2l1 /= max_sin
    # H_l /= max_sin

    #Pinv = np.diag(1. / s).dot(U.T)
    Pinv = np.diag(1. / (s**0.5)) @ U.T
    #Pinv = np.linalg.pinv(U @ np.diag(s))
    #print(Pinv)
    #print('singular values', s)
    Sinv = V.T @ np.diag(1./ (s**0.5))

    A = np.tensordot(Pinv, H_2l1.reshape([d ** l, d, d ** l * out_dim]), axes=(1, 0))
    A = np.tensordot(A, Sinv, axes=[2, 0])
    h_l = H_l.ravel()
    if out_dim == 1:
        omega = Pinv.dot(h_l)
        #print(max(h_l))
    else:
        omega = (Pinv.dot(H_l.reshape([d**l,out_dim])))
    alpha = Sinv.T.dot(h_l)
    # alpha = U[0, :]
    # omega = V[:, 0]
    model = LinRNN(alpha, A, omega)
    return model
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
# print(Y.shape)

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def OLS(X,Y):
    '''
    OLS method to recover the tensor
    :param X: Input data, of dimension n*l*d_x
    :param Y: Output data, of dimension n*d_y
    :return: recovered tensor
    '''
    if Y.ndim == 1:
        Y = Y.reshape((Y.shape[0], 1))

    N = len(X)
    dim = X.shape[1]
    l = X.shape[2]
    p = Y.shape[1]
    temp  = X[:, :, 0]
    #print(X.shape)
    for i in range(1, X.shape[2]):
        temp = khatri_rao(temp, X[:, :, i])
    X = temp
    #print(X.shape)
    return np.linalg.lstsq(X.reshape(N,dim**l),Y, rcond=None)[0].reshape([dim]*l + [p])

import pollution
normalize = True
exp = 'Pollution'
length = 3
n_dim = 1
noise_level = 0.1
data, scaler = pollution.prepare_data(add_bias=True, normalize=normalize)
data_size = len(data)
training_data = data[:int(data_size/10)]
#training_data = shuffle(training_data)
test_data = data[int(data_size/10):int(data_size/10)+720]
padded = True
n_epochs =1
le = [2, 4, 5]
test_les = [3]
ranks = [5]
inner_ranks = [2, 3, 4, 5, 6, 7, 8]
out_dim = 1
num_examples = 4000
all_num_examples = 4000
use_init = True
for rank in ranks:
    tt_hankel = []
    for length in le:
        if not padded:
            if exp == 'Pollution':
                data_function = lambda l, n: pollution.generate_pollution_data(training_data, l, out_dim=out_dim, num_examples=n)
                data_function_test = lambda l: pollution.generate_pollution_data(test_data, l, out_dim=out_dim)
            else:
                data_function = lambda l, n: generate_simple_addition(num_examples=n, traj_length=l, n_dim=n_dim,
                                                noise_level=noise_level)
                data_function_test = lambda l: generate_simple_addition(num_examples=1000, traj_length=l,
                                                        n_dim=n_dim, noise_level=noise_level)


            Xtest, ytest = data_function_test(l=length)

            X, Y = data_function(n = all_num_examples, l = length)

            X, Y = shuffle(X, Y)
            X = X[:num_examples]
            Y = Y[:num_examples]


            if use_init:
                init_tt = recover_tt_from_file('init_tt_'+str(X.shape[2]))
            else:
                init_tt = None
            train_error_ALS, test_error_ALS, tensor_train = train_ALS(X, Y, rank, Xtest, ytest, n_epochs, init_tt)
            tt = convert_to_array(tensor_train)
            #tt = tt_SVD(tt)
            tt = recover_tensor(tt).squeeze()
            tt_hankel.append(tt)
            #print('current tt:', tt.shape)

    if padded:
        length = le[0]
        import synthetic_data
        if exp == 'Pollution':
            data_function = lambda l, n: pollution.generate_pollution_data(training_data, l, out_dim=out_dim,
                                                                           num_examples=n)
            data_function_test = lambda l: pollution.generate_pollution_data(test_data, l, out_dim=out_dim)
        else:
            data_function = lambda l, n: generate_simple_addition(num_examples=n, traj_length=l, n_dim=n_dim,
                                                                  noise_level=noise_level)
            data_function_test = lambda l: generate_simple_addition(num_examples=1000, traj_length=l,
                                                                    n_dim=n_dim, noise_level=noise_level)
        X = []
        Y = []
        for i in range(length * 2 + 2):
            tempx, tempy = data_function(i, num_examples)

            X.append(tempx)
            Y.append(tempy)
        tt_hankel = []
        temp_index_list = [length, 2 * length, 2 * length + 1]
        for i in temp_index_list:
            X_temp_vec = []
            Y_temp_vec = []
            for j in range(1, i + 1):
                X_temp = X[j]
                Y_temp = Y[j].reshape(-1, 1)
                X_temp_vec.append(X_temp)
                Y_temp_vec.append(Y_temp)
            X_temp, Y_temp = synthetic_data.pad_data(X_temp_vec, Y_temp_vec)
            # Xtest, ytest = data_function_test(l=temp_index_list[i])
            # Xtest, ytest = synthetic_data.pad_data([Xtest], [ytest])
            if use_init:
                init_tt = recover_tt_from_file('init_tt_'+str(X_temp.shape[2]))
            else:
                init_tt = None

            train_error_ALS, test_error_ALS, tensor_train = train_ALS(X_temp, Y_temp, rank, n_epochs = n_epochs, init_tt=init_tt)
            # for k in range(len(tensor_train)):
            #     print(tensor_train[k].shape)
            tensor_train = convert_to_array(tensor_train)
            H_temp = recover_tensor(tensor_train).squeeze()
            H_temp_shape = list(H_temp.shape)
            H_temp_shape.append(1)
            H_temp = H_temp.reshape(tuple(H_temp_shape))
            tt_hankel.append(H_temp)

    for inner_rank  in inner_ranks:

        if inner_rank > rank:
            #inner_rank = rank
            continue
        else:
            print('current rank is:', rank)
            #rank = inner_rank
            print('current inner rank is:', inner_rank)

        try:
            RNN = spectral_learning(inner_rank, tt_hankel[1], tt_hankel[2], tt_hankel[0])
        except:
            print('Singular values too small for spectral learning')
            continue

        for test_le in test_les:

            if exp == 'Pollution':
                data_function_test = lambda l: pollution.generate_pollution_data(test_data, l, out_dim= out_dim)
            else:
                data_function_test = lambda l: generate_simple_addition(num_examples=1000, traj_length=l,
                                                                        n_dim=n_dim, noise_level=noise_level)
            Xtest, ytest = data_function_test(l=test_le)
            if padded:
                Xtest, ytest = synthetic_data.pad_data([Xtest], [ytest])
            Xtest = np.swapaxes(Xtest, 1, 2)
            # print(Xtest.shape)
            # print(RNN.alpha)
            # print(RNN.Omega)
            # print(RNN.A)
            # max_sin_val = 0
            # for i in range(RNN.A.shape[1]):
            #     U, D, V = np.linalg.svd(RNN.A[:, i, :])
            #     if max_sin_val < D[0]:
            #         max_sin_val = D[0]
            #RNN.A = RNN.A/max_sin_val
            #print('singular', max_sin_val)
            # print(RNN.Omega)
            # print(RNN.alpha)
            #RNN.alpha = RNN.alpha.reshape(1, rank)
            #print(Xtest,.s)
            pred = []
            #print(Xtest.shape)
            #Xtest = np.swapaxes(Xtest, 1, 2)
            for o in Xtest:
                pred.append(RNN.predict(o).ravel())
            pred = np.asarray(pred)
            # print(pred[150:160])
            # print(ytest[150:160])
            error = np.abs(pred - ytest)




            print('test le is', test_le)
            if not normalize:
                print('MAE:', np.mean(error))
                print('RMSE:', np.mean((pred - ytest)**2)**0.5)
                print('MAPE:', mean_absolute_percentage_error(pred, ytest))
            #print()
            #print('here', RNN.predict(Xtest[0]), ytest[0])

            else:
                pred = np.asarray(pred).reshape(ytest.shape)
                pred = (pred * (scaler.var_[4]**0.5) + scaler.mean_[4])
                ytest = (ytest * (scaler.var_[4]**0.5) + scaler.mean_[4])
                print(pred[0:3], ytest[0:3])
                from matplotlib import pyplot as plt

                plt.plot(pred[:100], c='r')
                plt.plot(ytest[:100], c='b')
                plt.show()
                error = np.abs(pred - ytest)
                print('MAE:', np.mean(error))
                print('RMSE:', np.mean((pred - ytest) ** 2) ** 0.5)
                print('MAPE:', mean_absolute_percentage_error(pred, ytest))



#print(tt)
#print(tensor_train)
#
#X, Y = (torch.from_numpy(X)).float(), (torch.from_numpy(Y)).float()
# X_val, Y_val = (torch.from_numpy(X_val)).float(), (torch.from_numpy(Y_val)).float()
# input_dim = X.shape[1]
# output_dim = Y.reshape(len(Y), -1).shape[1]
# print('starting')
# model = Net(rank, input_dim, output_dim, traj_length).to(device)
# print('starting')
# criterion = nn.MSELoss()
# lr = 0.02
# optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
# batch_size = 256
# n_epochs = 10000
# train_error_GD, test_error_GD, model = train_gradient_descent_standard(X, Y, X_val, Y_val, model, optimizer, criterion, n_epochs, device, batch_size)