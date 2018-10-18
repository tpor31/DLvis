from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None

    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    #x = x.reshape((-1,np.prod(x.shape[1:])))
    t = x
    out = np.dot(t.reshape((-1,np.prod(x.shape[1:]))),w) + b

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################

    dx = (np.dot(dout, w.T)).reshape((x.shape))
    z = x.copy()
    z = z.reshape((-1,np.prod(x.shape[1:])))
    dw = np.dot(z.T,dout)
    db = np.sum(dout, axis=0)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(0,x)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################

    dx = dout.copy()
    dx[x <= 0] = 0
    #dx = dout * (x > 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #############################################################################
        # TODO: [OURS] We have implemented for you the training-time forward pass   #
        # for batch normalization.                                                  #
        # Use minibatch statistics to compute the mean and variance, use these      #
        # statistics to normalize the incoming data, and scale and shift the        #
        # normalized data using gamma and beta.                                     #
        #                                                                           #
        # We need to store the output in the variable out. Any intermediates that   #
        # are needed for the backward pass are stored in the cache variable.        #
        #                                                                           #
        # We use the computed sample mean and variance together with                #
        # the momentum variable to update the running mean and running variance,    #
        # storing the result in the running_mean and running_var variables.         #
        #############################################################################
        
        # sample_mean = np.mean(x,axis=0,keepdims=True)
        # sample_var = np.var(x, axis=0, keepdims=True, ddof=1)
        #
        # x -= sample_mean
        # x /= (np.sqrt(sample_var)+eps)
        #
        # running_mean = momentum * running_mean + (1 - momentum)*sample_mean
        # running_var = momentum * running_var + (1 - momentum) * sample_var
        #
        # x *= gamma
        # x += beta
        #
        # out = x
        
        # FORWARD PASS: Step-by-Step
        
        # Step 1. m = 1 / N \sum x_i
        m = np.mean(x, axis=0, keepdims=True)
        
        # Step 2. xc = x - m
        xc = x - m
        
        # Step 3. xc2 = xc ^ 2
        xcsq = xc ** 2
        
        # Step 4. v = 1 / N \sum xc2_i
        v = np.mean(xcsq, axis=0, keepdims=True)
        
        # Step 5. vsq = sqrt(v + eps)
        vsqrt = np.sqrt(v + eps)
        
        # Step 6. invv = 1 / vsq
        invv = 1.0 / vsqrt
        
        # Step 7. xn = xc * invv
        xn = xc * invv
        
        # Step 8. xg = xn * gamma
        xgamma = xn * gamma
        
        # Step 9. out = xg + beta
        out = xgamma + beta
        
        cache = (x, xc, vsqrt, v, invv, xn, gamma, eps)
        
        running_mean = momentum * running_mean + (1 - momentum) * m
        running_var = momentum * running_var + (1 - momentum) * v
        #######################################################################
        #                           END OF [OUR] CODE                         #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        xc = x - bn_param['running_mean']
        vsqrt = np.sqrt(bn_param['running_var'] + eps)
        invv = 1.0 / vsqrt
        xn = xc * invv
        
        xgamma = xn * gamma
        out = xgamma + beta
        
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ##############################################################################
    # TODO: [OURS] We have implemented the backward pass for batch normalization.#
    # Results are stored in the dx, dgamma, and dbeta variables.                 #
    ##############################################################################
    (x, xc, vsqrt, v, invv, xn, gamma, eps) = cache
  
    N, D = x.shape
  
    # BACKWARD PASS: Step-byStep
  
    # Step 9. out = xg + beta
    dxg = dout
    dbeta = np.sum(dout, axis=0)
  
    # Step 8. xg = xn * gamma
    dxn = dxg * gamma
    dgamma = np.sum(dxg * xn, axis=0)
  
    # Step 7. xn = xc * invv
    dxc1 = dxn * invv
    dinvv = np.sum(dxn * xc, axis=0)
  
    # Step 6. invv = 1 / vsqrt
    dvsqrt = -1 / (vsqrt ** 2) * dinvv
  
    # Step 5. vsqrt = sqrt(v + eps)
    dv = 0.5 * dvsqrt / np.sqrt(v + eps)
  
    # Step 4. v = 1 / N \sum xcsq_i
    dxcsq = 1.0 / N * np.ones((N, D)) * dv
  
    # Step 3. xcsq = xc ^ 2
    dxc2 = 2.0 * dxcsq * xc
  
    # Step 2. xc = x - m
    dx1 = dxc1 + dxc2
    dm = - np.sum(dxc1 + dxc2, axis=0, keepdims=True)
  
    # Step 1. m = 1 / N \sum x_i
    dx2 = 1.0 / N * np.ones((N, D)) * dm
  
    dx = dx1 + dx2
  
    #############################################################################
    #                             END OF [OUR] CODE                             #
    #############################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
                
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
    
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
    
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask
    
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    p = conv_param['pad']
    s = conv_param['stride']
    
    assert (H + 2 * p - HH) % s == 0
    assert (W + 2 * p - HH) % s == 0
    
    out_height = (H + 2 * p - HH) // s + 1
    out_width = (W + 2 * p - WW) // s + 1

    #x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    
    x_pad = np.zeros((N,C,H+2*p,W+2*p))
    for n in range(N):
        for c in range(C):
            x_pad[n,c] = np.pad(x[n,c],(1,1),'constant', constant_values=(0,0))
    """
    out = np.zeros((N,F,out_height,out_width))
    for n in range(N):
        for f in range(F):
            for out_h in range(out_height):
                for out_w in range(out_width):
                    current_image = x_pad[n,:,out_h:out_h+HH,out_w*s:WW+s*out_w]
                    current_filter = w[f]
                    out[n,f,out_h,out_w] = np.sum(current_image*current_filter)
                    out[n,:,out_h,out_w] += b
                  
      """
    out = np.zeros((N,F,out_height,out_width)) 
    for n in range(N):
        for oh in range(out_height):
            for ow in range(out_width):
                for f in range(F):
                    current_img = x_pad[n,:, oh*s: oh*s+HH, ow*s:ow*s+WW]
                    current_filter = w[f] 
                    out[n,f,oh,ow] = np.sum(current_img*current_filter)
                out[n,:,oh,ow] += b
            
            
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape

    pad = conv_param['pad']
    stride = conv_param['stride']
    x_with_pad = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), 'constant', constant_values=0)

    N, F, Hdout, Wdout = dout.shape

    H_out = 1 + (H + 2 * conv_param['pad'] - HH) / conv_param['stride']
    W_out = 1 + (W + 2 * conv_param['pad'] - WW) / conv_param['stride']

    db = np.zeros((b.shape))
    for i in range(0, F):
        db[i] = np.sum(dout[:, i, :, :])

    dw = np.zeros((F, C, HH, WW))
    for i in range(0, F):
        for j in range(0, C):
          for k in range(0, HH):
            for l in range(0, WW):
                dw[i, j, k, l] = np.sum(dout[:, i, :, :] * x_with_pad[:, j, k:k + Hdout * stride:stride, l:l + Wdout * stride:stride])

    dx = np.zeros((N, C, H, W))
    for nprime in range(N):
        for i in range(H):
            for j in range(W):
                for f in range(F):
                    for k in range(Hdout):
                        for l in range(Wdout):
                            mask1 = np.zeros_like(w[f, :, :, :])
                            mask2 = np.zeros_like(w[f, :, :, :])
                            if (i + pad - k * stride) < HH and (i + pad - k * stride) >= 0:
                                mask1[:, i + pad - k * stride, :] = 1.0
                            if (j + pad - l * stride) < WW and (j + pad - l * stride) >= 0:
                                mask2[:, :, j + pad - l * stride] = 1.0

                            w_masked = np.sum(w[f, :, :, :] * mask1 * mask2, axis=(1, 2))
                            dx[nprime, :, i, j] += dout[nprime, f, k, l] * w_masked


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width'] 
    s = pool_param['stride']
    N, C, H, W = x.shape
    xm = x.copy()
    out = np.zeros((N,C,H//2,W//2)) 
    for n in range(N):
        for h in range(H//2):
            for w in range(W//2):
                for c in range(C):
                    current_pool = xm[n,c,h*s:h*s+pool_height,w*s:w*s+pool_width]
                    out[n,c,h,w] = np.max(current_pool)                
                

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width'] 
    s = pool_param['stride']
    N, C, H_D, W_D = dout.shape
    dx = np.zeros(x.shape)
    for n in range(N):
        for c in range(C):
            for hd in range(H_D):
                for wd in range(W_D):
                    x_p = x[n,c,hd*s:hd*s+pool_height,wd*s:wd*s+pool_width]
                    mask = (x_p == np.max(x_p))
                    dx[n,c,hd*s:hd*s+pool_height,wd*s:wd*s+pool_width] =  mask * dout[n,c,hd,wd]
            
        
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    N, C, H, W = x.shape
    x_sn_bn = x.reshape((N*H*W),C)
    
    out, cache = batchnorm_forward(x_sn_bn, gamma, beta, bn_param)
    out = out.reshape(N, C, H, W)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    N, C, H, W = dout.shape
    dout_sn_bn = dout.reshape((N*H*W),C)
    
    dx, dgamma, dbeta = batchnorm_backward(dout_sn_bn, cache)
    dx = dx.reshape(N, C, H, W)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = (H + 2 * padding - field_height) / stride + 1
    out_width = (W + 2 * padding - field_width) / stride + 1

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                                 stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols
