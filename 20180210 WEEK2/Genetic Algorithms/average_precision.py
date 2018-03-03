# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 13:22:00 2016

https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
@author: benhamner
"""

import numpy as np

def apk(actual, predicted, k=7):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
#    print 'actual',actual
#    print 'predicted',predicted
    #return 0 if there's no element for predict
    if len(actual) == 0:
        return 0.0
    #crop the prediction if necessary
    if len(predicted)>k:
        predicted = predicted[:k]
    #Compute the apk
    score = 0.0
    num_hits = 0.0
    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    return score / min(len(actual), k)

def mapk(actual, predicted, k=7):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])
#    if type(actual) is list:
#        return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])
#    else:
#        apk_array = np.zeros(len(actual))
#        for i in range(len(actual)):
#            apk_array[i] = apk(actual[i], predicted[i], k)
#        return np.mean(apk_array)