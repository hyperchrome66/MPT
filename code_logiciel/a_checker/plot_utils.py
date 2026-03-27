 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 14:58:37 2025

@author: Paolo Pierobon paolo.pierobon@inserm.fr
"""

import matplotlib.pyplot as plt 
import numpy as np
import math


def plotECDF(*args):
    """
    Plots multiple vectors side by side if they are 1D.
    If a single matrix (2D array) is provided, it plots its columns separately.
    Parameters:
    *args : list of 1D arrays or a single 2D array
            Optionally, the last argument can be a list/tuple/set of legend labels.
    """
    
    # Check if last arg is a list of strings (for labels)
    labels = None
    if isinstance(args[-1], (list, tuple, set)) and all(isinstance(l, str) for l in args[-1]):
        labels = list(args[-1])
        args = args[:-1]  # Remove label list from data
    
    # define lambda function for plotting cdf
    plot_cdf = lambda x, labText: (lambda sx, cdf: plt.plot(sx, cdf, marker=".", label = labText))(
        np.sort(x[~np.isnan(x)]),
        np.arange(1, len(x[~np.isnan(x)]) + 1) / len(x[~np.isnan(x)])
    )
    #    plt.figure(figsize=(8, 5))  # Set figure size
    

     # Convert all inputs to numpy arrays
    processed_args = [np.asarray(arg) for arg in args]

    # If there's only one argument and it's a 2D array, plot its columns
    if len(processed_args) == 1 and processed_args[0].ndim == 2:
        matrix = processed_args[0]  # The given argument is a matrix
    else:
        # Handle multiple vectors (1D arrays of potentially different lengths)
        max_len = max(arg.shape[0] for arg in processed_args)  # Find max length
        matrix = np.full((max_len, len(processed_args)), np.nan)  # Initialize with NaN

        # Fill matrix column-wise
        for i, vec in enumerate(processed_args):
            matrix[:len(vec), i] = vec

    # Plot each column
    plt.gca()
    for i in range(matrix.shape[1]):
        label = labels[i] if labels and i < len(labels) else f'Column {i + 1}'
        plot_cdf(matrix[:, i], label)

    plt.xscale("log")
    plt.legend()
    plt.ylabel("CDF")
    plt.show()


def plotVarViolin(data, variable, logscale=0):
    alldata=[]
    q=[]
    for k in data:
        tracks=k["tracks"]
        alldata.append(np.array([tracks[variable] for tracks in tracks.values()]))
        q.append([0.25, 0.75])
       
    plt.violinplot(alldata,  quantiles= q, showmedians=True, showextrema=False)
    if logscale:
        plt.yscale("log")

    # Customize the plot
#    plt.xticks([1, 2], ['Group A', 'Group B'])
    plt.ylabel(variable)


def plot_confidence_interval(x, mat, label="Mean Curve", alpha=0.3):
    """Plots a mean curve with a shaded confidence interval, using the same color as the curve."""
    
    # compute the mean and the interval of confidence
    y_mean = np.nanmean(mat,0)
    numel=np.sum(~np.isnan(mat),0) # number of non nan elements
    y_sub = np.nanmean(mat,0)-np.nanstd(mat,0)/np.sqrt(numel)
    y_sup = np.nanmean(mat,0)+np.nanstd(mat,0)/np.sqrt(numel)
    # Plot the mean curve and get its color 
    line, = plt.plot(x, y_mean, label=label)
    color = line.get_color()  # Automatically get the line color
    
    # Shade the confidence interval using the same color
    plt.fill_between(x, y_sub, y_sup, color=color, alpha=alpha, label="Confidence Interval")


# make small plot of lines or column sof a matrix
def plot_matrix(m, d=0):
    if d==1:
        m=np.transpose(m)
    ndata=np.shape(m)[0]
    if ndata>100:
        ndata=100
    raws = math.ceil(math.sqrt(ndata))  # ceil round up to next integer
    fig, ax = plt.subplots(raws, raws)
    endcond=False
    for i in range(raws):
        for j in range(raws):
            if (i+raws*j)<ndata:
                 plt.sca(ax[i,j])
                 plt.plot(m[i+raws*j,:])
            else:
                endcond=True
                break
        if endcond:
            break
    
