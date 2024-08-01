# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 09:39:46 2023

@author: Pierre Labouré
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats
 


def DTW(s1, s2, mask):
    # Performs calculates the matrix of costs by DTW method
    # Uses specified mask to note calculate all
    
    # Source : https://en.wikipedia.org/wiki/Dynamic_time_warping

    N = s1.shape[0]
    M = s2.shape[0]
    DTW_mat = np.full((N, M), np.inf)
    
    DTW_mat[0, 0] = 0

    for i in range(1, N):
        for j in range(1, M):
            if mask[i, j]:
                cost = np.abs(s1[i]-s2[j])
                last_min = np.min([DTW_mat[i-1, j], DTW_mat[i, j-1], DTW_mat[i-1, j-1]])
                DTW_mat[i, j] = cost + last_min
    return DTW_mat


def derivative_estimation(sig):
    # Estimatione of derivative for DDTW
    
    # Source : Derivative Dynamic Time Warping
    # https://www.ics.uci.edu/~pazzani/Publications/sdm01.pdf

    derivative = np.empty(len(sig))
    derivative[1:-1] = (sig[1:-1] - sig[0: -2] + (sig[2:] - sig[0: -2])/2)/2
    derivative[0] = derivative[1]
    derivative[-1] = derivative[-2]
    return derivative


def DDTW(s1, s2, mask):
    # Calculate the cost matrix by the Derivative DTW technique

    N = s1.shape[0]
    M = s2.shape[0]
    DTW_mat = np.full((N, M), np.inf)

    d1 = derivative_estimation(s1)
    d2 = derivative_estimation(s2)

    DTW_mat[0, 0] = 0

    for i in range(1, N):
        for j in range(1, M):
            if mask[i, j]:
                cost = (d1[i] - d2[j])**2
                
                last_min = np.min([DTW_mat[i-1, j], DTW_mat[i, j-1], DTW_mat[i-1, j-1]])
                DTW_mat[i, j] = cost + last_min
    return DTW_mat


def DTW_couples(DTW_mat):
    # Outputs all couples using the input DTW cost matrix
    
    (i, j) = DTW_mat.shape
    i-=1
    j-=1

    DTW_couples = [(i, j)]

    while (i, j) != (0, 0):
        idx_link = np.argmin([DTW_mat[i-1, j], DTW_mat[i, j-1], DTW_mat[i-1, j-1]])
        if DTW_mat[i-1, j] == DTW_mat[i, j-1] == DTW_mat[i-1, j-1]:
            i-=1
            j-=1
        elif idx_link == 0:
            i-=1
        elif idx_link == 1:
            j-=1
        else:
            i-=1
            j-=1
        DTW_couples.append((i, j))
        #DTW_couples = DTW_couples[::-1]
        
    return DTW_couples


def DTW_show_matrix(DTW_matrix):
    # Plots the matrix of links between points of the 2 curves

    show_matrix = np.zeros(DTW_matrix.shape)
    couples = DTW_couples(DTW_matrix)

    for (i, j) in couples:
        show_matrix[i, j] = 1
        
    show_matrix[np.diag_indices(DTW_matrix.shape[0])] = 0.5
    
    plt.figure()
    plt.imshow(show_matrix)
    plt.title('DTW matrix')
    plt.show()


def DTW_plot_links(s1, s2, mask, method):
    # Plots both curves with the links between each points

    plt.figure(figsize = (15, 15))
    
    N = s1.shape[0]
    M = s2.shape[0]

    if method == 'regular': #Use DTW
        DTW_mat = DTW(s1, s2, mask)
        plt.title('DTW')
    elif method == 'derivative': #Use DDTW
        DTW_mat = DDTW(s1 ,s2, mask)
        plt.title('Derivative DTW')

    couples = DTW_couples(DTW_mat)

    plt.plot(np.arange(N), s1, color = 'blue', label = 's1')
    plt.plot(np.arange(M), s2, color = 'purple', label = 's2')
    

    for (i, j) in couples:
        plt.plot([i, j], [s1[i], s2[j]], color = 'red')

    plt.legend()
    plt.show()
    return DTW_mat


def sakoe(s1, s2, r):
    # Masks the cost matrix with a band shape
    # Allows to skip a lot of computing
    # r is the maximum difference between i and j index
    
    # Dynamic Time Warping: Itakura vs Sakoe-Chiba

    N = s1.shape[0]
    M = s2.shape[0]

    mask = np.full((N, M), True)

    for i in range(N):
        for j in range(M):
            if (j >= i + r) or (j <= i - r):
                mask[i, j] = False
    return mask


def DTW_histogram(couples):
    # Plots and outputs the histogram of difference between the indices of each couple

    N = max(couples[0])

    keys = np.arange(-N, N+1)

    dict_hist = {key:0 for key in keys}    
    for (i, j) in couples:
        diff = i - j
        dict_hist[diff] += 1

    offsets = []
    counts = []
    for key in dict_hist:
        offsets.append(key)
        counts.append(dict_hist[key])
    plt.bar(offsets, counts)
    plt.show()
    return dict_hist


def offset_histogram(s1, s2, DTW_mat):
    # Plots the histogram of offsets and the skewed normal fit
    
    plt.figure(figsize = (15, 15))
    
    couples = DTW_couples(DTW_mat)
    couples = np.array(couples)
    
    c = 20
    left, right = ROI_boundaries(s1, s2, c)
    
    ROI = (couples[:, 0]>=left) & (couples[:, 0]< right) & (couples[:, 1]>=left) & (couples[:, 1]< right)
    couples_ROI = couples[ROI]

    A = np.zeros((right-left, right-left))
    A[(couples_ROI - left)[:, 0], (couples_ROI - left)[:, 1]] = 1
    plt.imshow(A)
    plt.title('ROI DTW matrix')
    plt.show()
    
    thresh = max(np.max(s1), np.max(s2))/c
    plt.figure(figsize = (10, 10))
    plt.plot(s1, color = 'blue', label = 'spectrum 1')
    plt.plot(s2, color = 'blue', label = 'spectrum 2')
    plt.plot([left, left], [0, thresh*c], color = 'red', label = 'ROI boundaries')
    plt.plot([right, right], [0, thresh*c], color = 'red')
    plt.legend()
    plt.show()
    
    
    plt.figure(figsize = (10, 10))
    diffs = np.abs(couples_ROI[:, 0] - couples_ROI[:, 1])
    bins = np.arange(-50, 51)
    hist, edges = np.histogram(diffs, bins-1/2)
    plt.bar(bins[:-1], hist, label = 'shifts histogram')
    
    # https://stackoverflow.com/questions/50140371/scipy-skewnormal-fitting
    X = np.linspace(-50, 50, 1000)
    a, loc, scale = stats.skewnorm.fit(diffs)
    plt.plot(X, len(diffs) * stats.skewnorm.pdf(X, a, loc, scale), color = 'red', label = 'skewed normal distribution')
    
    plt.legend()
    plt.show()
    

def hist_fit_skewed(s1, s2, DTW_mat, c):
    # Fits a skewed normal distribution on the histogram of the offsets
    
    couples = DTW_couples(DTW_mat)
    couples = np.array(couples)
    
    left, right = ROI_boundaries(s1, s2, c)
    
    ROI = (couples[:, 0]>=left) & (couples[:, 0]< right) & (couples[:, 1]>=left) & (couples[:, 1]< right)
    couples_ROI = couples[ROI]
    
    diffs = np.abs(couples_ROI[:, 0] - couples_ROI[:, 1])
    a, loc, scale = stats.skewnorm.fit(diffs)
    
    return a, loc, scale
    
def ROI_boundaries(s1, s2, c):
    # https://stackoverflow.com/questions/10582795/finding-the-full-width-half-maximum-of-a-peak
    # finds the edges of a ROI define by a coefficient c.
    # We look only at the signals at indices where at least one of the signal is greater than
    # the maximum of s1 or s2 devided by c
    # Returns the left and right boundaries
    
    thresh = max(np.max(s1), np.max(s2))/c
    
    d1 = np.sign(thresh - s1[:-1]) - np.sign(thresh - s1[1:]).squeeze()
    d2 = np.sign(thresh - s2[:-1]) - np.sign(thresh - s2[1:]).squeeze()
    
    left1 = np.where(d1>0)[0][0]
    right1 = np.where(d1<0)[0][-1]
    
    left2 = np.where(d2>0)[0][0]
    right2 = np.where(d2<0)[0][-1]
    
    left = min(left1, left2)
    right = max(right1, right2)
    
    return left, right
    
    
def ROI_compare_amp(s1, s2, DTW_mat, c):
    # Compares the amplitudes of the 2 spectra based on the DTW matrix in the given ROI
    
    couples = DTW_couples(DTW_mat)
    couples = np.array(couples)
    
    left, right = ROI_boundaries(s1, s2, c)
    
    ROI = (couples[:, 0]>=left) & (couples[:, 0]< right) & (couples[:, 1]>=left) & (couples[:, 1]< right)
    couples_ROI = couples[ROI]
    
    R = []
    for couples in couples_ROI:
        r = s2[couples[1]]/s1[couples[0]]
        R.append(r)
    return np.mean(R)
    
    












