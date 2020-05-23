from matplotlib import pyplot as plt
import pickle
les = [2, 4, 5]
for le in les:
    init = pickle.load(open('init_tt_'+str(le), 'rb'))
    for i in range(len(init)):
        print(init[i], init[i].shape)