import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. You may need to modify some of the                #
  # code above to compute the gradient.                                       #
  #############################################################################

  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]

  for i in range(num_train):
    scores = X[i].dot(W)
    scores -=  np.max(scores) #To avoid numerical issues
    # Compute vector of scores
       
    # Compute loss (and add to it, divided later)
    p = lambda k: np.exp(scores[k]) / np.sum(np.exp(scores))
    #calcular la funcion de loss para el label y[i]
    loss += -np.log(p(y[i]))
    # Compute gradient
    # Here we are computing the contribution to the inner sum for a given i.
    for k in range(num_classes):
      #pk = softmax(scores[k], scores)
      pk = p(k)
      dW[:, k] += (pk - (k == y[i])) * X[i]

  loss /= num_train
  
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
    
  dW /= num_train  
  dW += reg * W  
   
 
  #pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

def softmax(z, s):
    return np.exp(z) / np.sum(np.exp(s))

def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  num_train = X.shape[0]
  num_classes = W.shape[1]  
  dW = np.zeros_like(W)
  
      
  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  scores = X.dot(W)  
  scores -=  np.max(scores) #To avoid numerical issues
  sj = np.sum(np.exp(scores), axis=1, keepdims=True)
  s = np.exp(scores) / sj 
  loss = np.sum(-np.log(s[np.arange(num_train),y]))  
    
  #oh = one_hot(y,num_classes)
    
  #dW = np.dot(X.T, (s - oh))  
  #derivada de funcion softmax  
  s[np.arange(num_train),y] -= 1
  dW = np.dot(X.T, s)

  loss /= num_train
  dW /= num_train     
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
    

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

