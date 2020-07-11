from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange
import math

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
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    y_pred = np.dot(X,W)  #A numpy array of shape (N,C) containing scores for each class for each x in the minibatch
    
    
    N = y_pred.shape[0]
    C = W.shape[1]
    for i in range (N):
        curr_scores = y_pred[i] #1-D Array of all scores for a particular test sample
        curr_scores = curr_scores-np.max(curr_scores)
        
        softmax = np.exp(curr_scores)/np.sum(np.exp(curr_scores))
        p = softmax[y[i]]
        
        loss += -1*math.log(p)*(1/N)
        softmax[y[i]] -= 1
        for j in range(C):
            dW[:,j] += X[i]*softmax[j]*(1/N)
            
               
    loss += reg*(np.sum(W*W))
    dW += reg*2*W
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    y_pred = np.dot(X,W)  #A numpy array of shape (N,C) containing scores for each class for each x in the minibatch
    
    
    N = y_pred.shape[0]
    C = W.shape[1]
    
    #normalizing the predictions in each row 
    y_pred = y_pred-np.max(y_pred,axis=1)[:,np.newaxis]
    exp_y_pred = np.exp(y_pred)
    
    softmax = exp_y_pred/np.sum(exp_y_pred,axis=1)[:,np.newaxis]#Matrix with softmax for each prediction as each row
    loss = np.sum(-np.log(softmax[np.arange(N),y]))*(1/N)#Add log of appropriate softmaxes to loss
    
    modified_softmax = softmax
    modified_softmax[np.arange(N),y] -=1
    
    dW = np.dot(X.transpose(),modified_softmax)*(1/N)
    
    loss += reg*np.sum(W*W)
    
    dW += reg*2*W
    
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
