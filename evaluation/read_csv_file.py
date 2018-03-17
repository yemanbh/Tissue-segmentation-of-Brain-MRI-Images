#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 10:33:43 2018

@author: yb
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_history(data, legend):
#    x  = range(1,len(poly_degree)+1)
    X = np.linspace(1, data.shape[0], data.shape[0], endpoint=True)

    for col_num, lgnd in enumerate(legend):
        Y  = data[:,col_num]
        plt.plot(X, Y, label=lgnd) 
        
    plt.legend()
    plt.show()

if __name__ == '__main__':
    
    training_history = pd.read_csv("depth5_patch32.csv")
    col_name  = list(training_history.columns)
    values  = training_history.values
    
    plot_history(values[:,1:7], col_name[1:7])
    
