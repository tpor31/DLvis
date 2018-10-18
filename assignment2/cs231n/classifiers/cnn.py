from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32, use_batchnorm=False):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.use_batchnorm = use_batchnorm
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        C, H, W = input_dim
        conv_size = filter_size*filter_size*C*num_filters
        
        self.params['W1'] = weight_scale * np.random.randn(num_filters, C ,filter_size, filter_size)# * np.sqrt(input_dim)
        self.params['b1'] = np.zeros(num_filters)
        input_dim_fc = num_filters*(H//2)*(W//2)
        
        self.params['W2'] = weight_scale * np.random.randn(input_dim_fc, hidden_dim) * np.sqrt(2/input_dim_fc)
        self.params['b2'] = np.zeros(hidden_dim)
        
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes) * np.sqrt(2/hidden_dim)
        self.params['b3'] = np.zeros(num_classes)
        
        if (use_batchnorm):
            self.params['gamma1'] = np.ones(C)
            self.params['beta1'] = np.zeros(C)
            self.params['gamma2'] = np.ones(input_dim_fc)
            self.params['beta2'] = np.zeros(input_dim_fc)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        
        mode = 'test' if y is None else 'train'
        if self.use_batchnorm:
            bn_param['mode'] = mode
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        if (self.use_batchnorm):
            gamma1, beta1 = self.params['gamma1'], self.params['beta1']
            gamma2, beta2 = self.params['gamma2'], self.params['beta2']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        
        if (self.use_batchnorm):
            out1, cache1 = conv_bn_relu_pool_forward(X, w1, b1, gamma, beta, conv_param, bn_param, pool_param)
            out2, cache2 = affine_bn_relu_forward(out1,gamma2,beta2, W2, b2, bn_param)
        
        else:
            out1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
            out2, cache2 = affine_relu_forward(out1, W2, b2)
        
       
        scores, cache3 = affine_forward(out2, W3, b3)
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, dout = softmax_loss(scores, y)
        loss += 0.5 * self.reg  * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
        
        dx, dw3, db3 = affine_backward(dout, cache3)
        grads['W3'] = dw3
        grads['b3'] = db3
        dx, dw2, db2 = affine_relu_backward(dx, cache2)
        grads['W2'] = dw2
        grads['b2'] = db2   
        dx, dw1, db1 = conv_relu_pool_backward(dx, cache1)
        grads['W1'] = dw1
        grads['b1'] = db1
        
         #regularitation
        grads['W3'] += self.reg * W3    
        grads['W2'] += self.reg * W2
        grads['W1'] += self.reg * W1
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
    
    def conv_bn_relu_pool_forward(x, w, b, gamma, beta, conv_param, bn_param, pool_param):
        a, conv_cache = conv_forward_fast(x, w, b, conv_param)
        an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
        rf, relu_cache = relu_forward(an)
        out, pool_cache = max_pool_forward_fast(rf, pool_param)
        cache = (conv_cache, bn_cache, relu_cache, pool_cache)
        return out, cache
    
    def affine_bn_relu_forward(x,gamma,beta, w, b, bn_param):
        a, fc_cache = affine_forward(x, w, b)
        b, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
        out, relu_cache = relu_forward(b)
        cache = (fc_cache, bn_cache, relu_cache)
        return out, cache
        
