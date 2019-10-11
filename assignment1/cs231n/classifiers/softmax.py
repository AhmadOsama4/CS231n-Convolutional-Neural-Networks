import numpy as np
from random import shuffle
from past.builtins import xrange

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
  num_train = X.shape[0]
  num_classes = W.shape[1]
  ##################################################################for c in range###########
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  #shift values
  scores -= np.max(scores, axis = 1).reshape(scores.shape[0], -1)
  scores_exp = np.exp(scores)

  scores_exp_norm = scores_exp / np.sum(scores_exp, axis=1).reshape(scores_exp.shape[0], -1)

  for i in range(num_train):
    loss += -scores[i][y[i]] + np.log( np.sum(scores_exp[i]) )
    # Correct class derivative: X(-1 + exp(yi)/Sum)
    scores_exp_norm[i][y[i]] -= 1

    dW += (X[i].T.reshape(X.shape[1], 1)) * (scores_exp_norm[i].reshape(1, num_classes))

  loss /= num_train
  dW /= num_train
  loss += reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  #shift values
  scores -= np.max(scores, axis = 1).reshape(scores.shape[0], -1)
  scores_exp = np.exp(scores)
  scores_exp_norm = scores_exp / np.sum(scores_exp, axis=1).reshape(scores_exp.shape[0], -1)

  loss = np.sum( -np.log( scores_exp_norm[np.arange(num_train), y] ) )
  scores_exp_norm[np.arange(num_train), y] -= 1
  dW += X.T.dot(scores_exp_norm)

  loss /= num_train
  dW /= num_train
  loss += reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

