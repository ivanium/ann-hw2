from network import Network
from layers import Relu, Linear, Conv2D, AvgPool2D, Reshape
from utils import LOG_INFO
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss
from solve_net import train_net, test_net
from load_data import load_mnist_4d

from plot import save_data

train_data, test_data, train_label, test_label = load_mnist_4d('data')

# Your model defintion here
# You should explore different model architecture
def basicConv2Layer():
    model = Network()
    model.add(Conv2D('conv1', 1, 4, 3, 1, 1))
    model.add(Relu('relu1'))
    model.add(AvgPool2D('pool1', 2, 0))  # output shape: N x 4 x 14 x 14
    model.add(Conv2D('conv2', 4, 4, 3, 1, 1))
    model.add(Relu('relu2'))
    model.add(AvgPool2D('pool2', 2, 0))  # output shape: N x 4 x 7 x 7
    model.add(Reshape('flatten', (-1, 196)))
    model.add(Linear('fc3', 196, 10, 0.1))

    loss = SoftmaxCrossEntropyLoss(name='loss')
    return model, loss

def LeNet():
    model = Network()
    model.add(Conv2D('conv1', 1, 6, 5, 2, 1))
    model.add(Relu('relu1'))
    model.add(AvgPool2D('pool1', 2, 0))  # output shape: N x 6 x 14 x 14
    model.add(Conv2D('conv2', 6, 16, 5, 0, 0.1))
    model.add(Relu('relu2'))
    model.add(AvgPool2D('pool2', 2, 0))  # output shape: N x 16 x 5 x 5
    model.add(Reshape('flatten', (-1, 400)))
    model.add(Linear('fc1',400, 120, 0.1))
    model.add(Relu('relu3'))
    model.add(Linear('fc2', 120, 84, 0.1))
    model.add(Relu('relu4'))
    model.add(Linear('fc3', 84, 10, 0.1))

    loss = SoftmaxCrossEntropyLoss(name='loss')
    return model, loss
# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

config = {
    'learning_rate': 0.05,
    'weight_decay': 0.0,
    'momentum': 0.0,
    'batch_size': 100,
    'max_epoch': 100,
    'disp_freq': 50,
    'test_epoch': 5
}

model, loss = LeNet()
# model, loss = basicConv2Layer()

for epoch in range(config['max_epoch']):
    LOG_INFO('Training @ %d epoch...' % (epoch))
    train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])

    if epoch % config['test_epoch'] == 0:
        LOG_INFO('Testing @ %d epoch...' % (epoch))
        test_net(model, loss, test_data, test_label, config['batch_size'])

save_data("lenet")
