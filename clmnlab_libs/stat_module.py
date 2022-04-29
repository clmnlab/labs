#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 16:15:22 2020

@author: yoonseojin
"""

import numpy as np 
from scipy import stats
from Module import sj_higher_function

"""
LSE
"""
def LSE(f, x, args, responses):
    """
    compute the function which is (f(x; args) - response)^2

    :param f: estimator function ex) lambda x1, x2, a: x1 + x2 +2a
    :param x: description variables ex) [[1,2], [3,4]]
    :param args: argumetns of f ex) [1,2]
    :param responses: response values of data ex) [4]
    :return: scalar(LSE value)
    """
    if len(x) == len(responses):
        result = 0
        for i in range(0, len(responses)):
            value = LSE_one_data(f, x[i], args, responses[i])
            result += value
    else:
        Exception("len(args_list): %.1f == len(responses): %.1f not matched" % (len(x), len(responses)))
    return result

def LSE_one_data(f, x, args, response):
    """
    computer the function which is (f(args) - response)^2
    and only compute one data

    :param f: estimator function ex) lambda x1, x2, a: x1 + x2 +2a
    :param x: description variables ex) [1,2]
    :param args: values for appling f ex) [1]
    :param response: response values of data ex) 4
    :return: scalar(LSE value)
    """

    return (sj_higher_function.apply_function(f, np.append(x, args)) - response) ** 2

"""
require for computing
"""

def make_design_matrix(variables):
    design_matrix = np.ones(variables.shape[0])
    if len(variables.shape) > 1:
        for i in range(0, variables.shape[1]):
            v = variables[:,i]
            design_matrix = np.c_[design_matrix, v]
    else:
        design_matrix = np.c_[design_matrix, variables]
    return design_matrix

def hat_matrix(variables):
    X = make_design_matrix(variables)
    intermediate_process = np.linalg.inv(np.dot(X.transpose(), X)) # (X'X)^-1
    result = np.dot(np.dot(X, intermediate_process), X.transpose()) # X(X'X)^-1X'
    return result

"""
coefficient
"""
def coeff_variance(variables, variance_of_error):
    # variance_of_error is scalar
    
    design_matrix = make_design_matrix(variables)
    intermediate_process = np.linalg.inv(np.dot(design_matrix.transpose(), design_matrix))
    
    return np.sqrt(intermediate_process) * np.sqrt(variance_of_error)

def inverse_XTX(variables):
    design_matrix = make_design_matrix(variables)
    intermediate_process = np.linalg.inv(np.dot(design_matrix.transpose(), design_matrix))
    return intermediate_process

def find_reg_coeff(variables, response):
    # variables is list
    # response is list
    design_matrix = make_design_matrix(variables)
        
    intermediate_process = np.dot(np.linalg.inv(np.dot(design_matrix.transpose(), design_matrix)), design_matrix.transpose())
    result = np.dot(intermediate_process, response)
    return result
    
"""
estimation
"""
def est_linear_reg(variables, coeff):
    design_matrix = make_design_matrix(variables)
    return np.dot(design_matrix, coeff)

"""
error 
"""
def residual(response, est):
    return response - est

def est_standard_error(variables, response):
    # find s^2
    if len(variables.shape) > 1:
        p = variables.shape[1] + 1
    else:
        p = 2

    n = len(response)
    M = np.identity(variables.shape[0]) - hat_matrix(variables)

    return np.dot(np.dot(response, M), response) / (n - p)

# standard error of Bj^ = SE(Bj^)
def std_error_coeff_estimation(variables, response, intereset_variable_index):
    s2 = est_standard_error(variables, response)
    coeff_variance = inverse_XTX(variables)

    se_coeff_variance = np.sqrt(s2) * np.sqrt(coeff_variance)
    return se_coeff_variance[intereset_variable_index, intereset_variable_index]

"""
Sum of Square
"""
def SSE(response, est):
    return np.sum(np.square(response - est))

def MSE(response, est, number_of_variables):
    # number_of_variables is count of variable excluded one vector in design matrix
    sse = SSE(response, est)
    count_of_data = len(response)
    return sse / (count_of_data - (number_of_variables + 1))

def SSR(est, origin):
    return np.sum( np.square(est - sample_mean(origin)) )

def MSR(est, origin, number_of_variables):
    # number_of_variables is count of variable excluded one vector in design matrix
    return SSR(est, origin) / number_of_variables

def SST(data):
    return np.sum(np.square(data - sample_mean(data)))
    

"""
GOF(Good Of Fitness)
"""
def coef_of_determination(data, est):
    """
    find R square

    :param data: numpy array(response value)
    :param est: numpy array(hypothesized response value)
    :return: scalar(R square)
    """
    return SSR(est, data) / SST(data)

"""
ANOVA
"""
def ratio_F(response, est, number_of_variables):
    return MSR(est, response, number_of_variables) / MSE(response, est, number_of_variables)

"""
basic statistics
"""
def sample_mean(data):
    return sum(data) / len(data)

def sample_variance(data):
    return np.sum(np.square(data - sample_mean(data))) / (len(data) - 1)

"""
Confidence Interval 
"""
def bonferroni_simultaneous_confidence_interval(variables, response, confidence_level, intereset_variable_indexes):
    n = len(response)
    g = len(intereset_variable_indexes)
    alpha = 1 - confidence_level

    B_hat = find_reg_coeff(variables, response)
    p = len(B_hat)
    alpha_star = alpha / g # α* = α/g

    result = []
    for interest_variable_index in intereset_variable_indexes:
        interest_B_hat = B_hat[interest_variable_index]
        SE_B_hat = std_error_coeff_estimation(variables, response, interest_variable_index)
        val1_CI = interest_B_hat - (stats.t.ppf(alpha_star/2, n-p) * SE_B_hat)
        val2_CI = interest_B_hat + (stats.t.ppf(alpha_star/2, n - p) * SE_B_hat)

        result.append([min(val1_CI, val2_CI), max(val1_CI, val2_CI)])
    return result

"""
test statistics
"""
def t_p_value(t_stat, df, method = "two-sided"):
    """
    Get p-value from t stat
    
    :param t_stat: t-statistics
    :param df: degree of freedom(int)
    :param method: (string) ex) two_sided, less_than, greater_than
    
    return p-value
    """
    if method == "two-sided":
        return (1 - stats.t.cdf(t_stat, df = df))*2
    elif method == "less":
        return scipy.stats.t.cdf(t_stat, df = df)
    elif method == "greater":
        return (1 - stats.t.cdf(t_stat, df = df))
    
def t_critical_value(p_value, df, method = "two-sided"):
    """
    Get critical value from p-value
    
    :param p_value: p_value
    :param df: degree of freedom(int)
    :param method: (string) ex) two_sided, less_than, greater_than
    
    return p-value
    """
    if method == "two-sided":
        return np.abs(stats.t.ppf(p_value / 2, df = df))
    elif method == "less":
        return stats.t.ppf(p_value, df = df)
    elif method == "greater":
        return stats.t.ppf(1 - p_value, df = df)
        

    
    