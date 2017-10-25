import matplotlib as mpl
mpl.use("Pdf")
import matplotlib.pyplot as plt

import numpy as np

mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
# plt.xlabel('iter')
fig = plt.figure('loss', figsize = (10, 10))

loss_ax = fig.add_subplot(221)
loss_ax.set_title('train-iter-loss')
loss_ax.set_ylabel('loss')
loss_ax.set_xlabel('iter')
loss_ax.set_yscale('log')

acc_ax = fig.add_subplot(222)
acc_ax.set_title('train-iter-accuracy')
acc_ax.set_ylabel('acc')
acc_ax.set_xlabel('iter')
acc_ax.set_yscale('linear')

test_loss_ax = fig.add_subplot(223)
test_loss_ax.set_title('test-iter-loss')
test_loss_ax.set_ylabel('loss')
test_loss_ax.set_xlabel('epoch')
test_loss_ax.set_yscale('linear')

test_acc_ax = fig.add_subplot(224)
test_acc_ax.set_title('test-iter-accuracy')
test_acc_ax.set_ylabel('acc')
test_acc_ax.set_xlabel('epoch')
test_acc_ax.set_yscale('linear')

# plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9)
plt.tight_layout()

local_loss_list = []
local_acc_list = []
def plot_loss(loss, acc, name = 'loss', plot = False):
    global local_loss_list, local_acc_list
    local_loss_list.append(loss)
    local_acc_list.append(acc)
    if(plot):
        xlen = len(local_loss_list)
        x = np.linspace(0, xlen*50, xlen, endpoint=False)

        loss_ax.plot(x, local_loss_list, color = 'b')

        acc_ax.plot(x, local_acc_list, color = 'b')

        fig.savefig(name + '.png')

local_test_loss_list = []
local_test_acc_list = []
def plot_test_acc(loss, acc, name = 'loss', plot = False):
    # if(acc_list.size == 0):
    #     return
    global local_test_acc_list, local_test_loss_list
    local_test_acc_list.append(acc)
    local_test_loss_list.append(loss)
    if(plot):
        xlen = len(local_test_acc_list)
        x = np.linspace(0, xlen*5, xlen, endpoint=False)

        test_acc_ax.plot(x, local_test_acc_list, color = 'b')
        test_loss_ax.plot(x, local_test_loss_list, color = 'b')

        fig.savefig(name + '.png')

def plot_all(name = 'loss'):
    xlen = len(local_loss_list)
    x = np.linspace(0, xlen*50, xlen, endpoint=False)
    loss_ax.plot(x, local_loss_list, color = 'b')
    acc_ax.plot(x, local_acc_list, color = 'b')

    xlen = len(local_test_acc_list)
    x = np.linspace(0, xlen*5, xlen, endpoint=False)
    test_acc_ax.plot(x, local_test_acc_list, color = 'b')
    
    # xlen = len(local_vali_acc_list)
    # x = np.linspace(0, xlen*50, xlen, endpoint=False)
    # vali_ax.plot(x, local_vali_acc_list, color = 'b')

    fig.savefig(name + '.png')
    save_data()

def save_data(name = ''):
    np.save(name + '_loss_list.npy', (local_loss_list))
    np.save(name + '_acc_list.npy', (local_acc_list))
    np.save(name + '_test_acc_list.npy', (local_test_acc_list))
    np.save(name + '_test_loss_list.npy', (local_test_loss_list))


def cal():
    fig = plt.figure('comp', figsize=(10,10))
    loss_ax = fig.add_subplot(211)
    loss_ax.set_title('train-iter-loss')
    loss_ax.set_ylabel('loss')
    loss_ax.set_xlabel('iter')
    loss_ax.set_yscale('log')

    acc_ax = fig.add_subplot(212)
    acc_ax.set_title('train-iter-accuracy')
    acc_ax.set_ylabel('acc')
    acc_ax.set_xlabel('iter')
    acc_ax.set_yscale('linear')
    plt.tight_layout()

    name = 'rr1_'
    loss_list0 = np.load(name + 'loss_list.npy')
    acc_list0 = np.load(name + 'acc_list.npy')

    name = 'rr01_'
    loss_list3 = np.load(name + 'loss_list.npy')
    acc_list3 = np.load(name + 'acc_list.npy')

    name = 'rr001_'
    loss_list6 = np.load(name + 'loss_list.npy')
    acc_list6 = np.load(name + 'acc_list.npy')

    name = 'rr108_'
    loss_list9 = np.load(name + 'loss_list.npy')
    acc_list9 = np.load(name + 'acc_list.npy')

    name = 'rr1010_'
    loss_list10 = np.load(name + 'loss_list.npy')
    acc_list10 = np.load(name + 'acc_list.npy')

    xlen = len(loss_list0)
    x = np.linspace(0, xlen*50, xlen, endpoint=False)
    loss_ax.plot(x, loss_list0, color = 'b', label = 'learning_rate = 0.1')
    loss_ax.plot(x, loss_list3, color = 'g', label = 'learning_rate = 0.01')
    loss_ax.plot(x, loss_list6, color = 'r', label = 'learning_rate = 0.001')
    # loss_ax.plot(x, loss_list9, color = 'orange', label = 'learning_rate = 0.')
    # loss_ax.plot(x, loss_list10, color = 'pink', label = 'learning_rate = 0.0')
    acc_ax.plot(x, acc_list0, color = 'b', label = 'learning_rate = 0.1')
    acc_ax.plot(x, acc_list3, color = 'g', label = 'learning_rate = 0.01')
    acc_ax.plot(x, acc_list6, color = 'r', label = 'learning_rate = 0.001')
    # acc_ax.plot(x, acc_list9, color = 'orange', label = 'hidden_layer_size = 100, 80')
    # acc_ax.plot(x, acc_list10, color = 'pink', label = 'hidden_layer_size = 100, 100')

    loss_ax.legend(loc = 'upper right')
    acc_ax.legend(loc = 'lower right')

    fig.savefig('comp.png')
    
if __name__ == '__main__':
    cal()
