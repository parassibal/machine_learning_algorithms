
import numpy as np
from numpy.core.numeric import identity
import pandas as pd

############################################################################
# DO NOT MODIFY CODES ABOVE 
# DO NOT CHANGE THE INPUT AND OUTPUT FORMAT
############################################################################

###### Part 1.1 ######
def mean_square_error(w, X, y):
    """
    Compute the mean square error of a model parameter w on a test set X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test features
    - y: A numpy array of shape (num_samples, ) containing test labels
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean square error
    """
    #####################################################
    # TODO 1: Fill in your code here                    #
    #####################################################
    mat_prod=np.matmul(X,w)
    sub_val=np.subtract(mat_prod,y)
    sq=np.square(sub_val)
    err=np.mean(sq)
    return(err)

###### Part 1.2 ######
def linear_regression_noreg(X, y):
    """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing features
  - y: A numpy array of shape (num_samples, ) containing labels
  Returns:
  - w: a numpy array of shape (D, )
  """
  #####################################################
  #	TODO 2: Fill in your code here                    #
  ######################################################

    x_trans=np.transpose(X)
    x_dot=np.dot(x_trans,X)
    y_dot=np.dot(x_trans,y)
    x_inv=np.linalg.inv(x_dot)
    w=np.dot(x_inv,y_dot)
    return(w)

###### Part 1.3 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing features
    - y: A numpy array of shape (num_samples, ) containing labels
    - lambd: a float number specifying the regularization parameter
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here                    #
  #####################################################	

    x_trans=np.transpose(X)
    x_dot=np.dot(x_trans,X)
    y_dot=np.dot(x_trans,y)
    size_x=np.size(X,axis=1)
    mat=np.identity(size_x)
    identity_matrix=lambd*mat
    x_identity=np.add(x_dot,identity_matrix)
    x_inv=np.linalg.inv(x_identity)
    w=np.dot(x_inv,y_dot)
    return(w)

###### Part 1.4 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training features
    - ytrain: A numpy array of shape (num_training_samples, ) containing training labels
    - Xval: A numpy array of shape (num_val_samples, D) containing validation features
    - yval: A numpy array of shape (num_val_samples, ) containing validation labels
    Returns:
    - bestlambda: the best lambda you find among 2^{-14}, 2^{-13}, ..., 2^{-1}, 1.
    """
    #####################################################
    # TODO 5: Fill in your code here                    #
    #####################################################		
    bestlambda=None
    lambd=[]
    initial_error=float('inf')
    for i in range(-14,1):
        lambd.append(2**(-i))
    for i in lambd:
        model=regularized_linear_regression(Xtrain,ytrain,i)
        error=mean_square_error(model,Xval,yval)
        if(initial_error>error):
            bestlambda=i
            initial_error=error
    return(bestlambda)
    

###### Part 1.6 ######
def mapping_data(X, p):
    """
    Augment the data to [X, X^2, ..., X^p]
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training features
    - p: An integer that indicates the degree of the polynomial regression
    Returns:
    - X: The augmented dataset. You might find np.insert useful.
    """
    #####################################################
    # TODO 6: Fill in your code here                    #
    #####################################################		
    mapped_data=[]
    for i in range(1,p+1):
        val=np.power(X,i)
        mapped_data.append(val)
    X=np.concatenate(mapped_data,axis=1)
    return(X)

"""
NO MODIFICATIONS below this line.
You should only write your code in the above functions.
"""
