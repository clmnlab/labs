
# https://www.machinelearningplus.com/statistics/mahalanobis-distance/

# Common Libraries
import pandas as pd
from scipy.linalg import inv
import numpy as np
import scipy

# Custom Libraries

# Sources

def mahalanobis(x=None, data=None, cov=None):
    """
    Compute the Mahalanobis Distance between each row of x and the data
    
    :param x: vector of matrix of data with, say, p columns
    :param data: ndarray of the distribution from which Mahalanobis distance of each observation of x is be computed.
    :param cov: covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    # x - μ
    x_minus_mu = x - np.mean(data)

    # Comput covariance
    if not cov:
        cov = np.cov(data.values.T, rowvar=True)
    # inverse covariance
    inv_covmat = inv(cov)

    # D^2 = (x-m)^T * C^-1 * (x-m)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal.diagonal()

def manhattan_distance(x,y): 
    if len(x)==len(y):
        r = 0 
        for i in range(0, len(x)):
            r+=abs(x[i]-y[i])
        return r
    else:
        return "error"
    
def minkowski_distance(x, y, lam):
    r = 0
    for a, b in zip(x, y):
        r+=(abs(a-b))**lam
    return (r)**(1.0/lam)
    
    
####### Examples #######
if __name__ == "__main__":
    filepath = 'https://raw.githubusercontent.com/selva86/datasets/master/diamonds.csv'
    df = pd.read_csv(filepath).iloc[:, [0,4,6]]
    df.head()    
    df_x = df[["carat", "depth", "price"]].head(500)
    df_x["mahala"] = mahalanobis(x=df_x,
                                 data=df[["carat", "depth", "price"]])
    df_x.head()






