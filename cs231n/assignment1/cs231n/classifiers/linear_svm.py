import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    correct_num = 0
    for j in range(num_classes):
      if j == y[i]:
        continue
      
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        correct_num += 1
        loss += margin
        dW[:,j] += X[i,:]
    dW[:,y[i]] += (-1)* correct_num * X[i,:]
   
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /=num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW +=2* reg *W;

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  
        
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)
  # print(scores.shape)
  train_num = X.shape[0];
  caculate_score = scores[np.arange(0,train_num),y]
  # print(y.shape)  
  # print(correct_class_score.shape)
  scores = scores - np.reshape(caculate_score,(train_num,1))+ 1
  
  mask = np.zeros(scores.shape)
  mask = scores>0
  mask[np.arange(start=0,stop=train_num,step=1),y]=0;
  news_margin = scores * mask;
  
  loss = sum(sum(news_margin))
  loss /= X.shape[0]
  loss +=reg * np.sum(W * W)



  
  train_number = X.shape[0]
  dw_mask = np.zeros(scores.shape)
  dw_mask[scores>0] = 1
  dw_mask[np.arange(0,train_num,1),y]= 0
  dw_mask[np.arange(0,train_num,1),y] = -1* np.sum(mask,axis=1)
  dW = X.transpose().dot(dw_mask)
  dW /= X.shape[0]
  dW +=  2* reg *W;
  

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW

    
