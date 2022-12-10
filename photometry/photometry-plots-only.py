#AUTHOR: VICTORIA SEDWICK
#ADAPTED FROM NEUROfpMETRICS AND ILARIA CARTA (MATLAB)

#https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
#https://stackoverflow.com/questions/11190735/python-matplotlib-superimpose-scatter-plots

fp_file="D:\Photometry-Fall2022\Pup Block\Trial 1-PE - Copy\Ball\\3\\3_split2022-09-21T17_45_29.CSV"

project_home="D:\Photometry-Fall2022\Pup Block\Trial 1-PE"
project_id='pup_1'

fp_fps=20

pre_s=10
post_s=20

#Which roi/fiber signal do you want to plot (0-2)?
z=0

#PROCEED WITH CAUTION CHANGING ANYTHING AFTER THIS LINE
#MAKE SURE ALL PACKAGES ARE INSTALLED
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numpy import exp
import numpy as np
from scipy import stats
from numpy import mean
import os, tkinter, matplotlib.colors
from pathlib import Path

# from Tkinter import * #https://www.geeksforgeeks.org/how-to-create-a-pop-up-message-when-a-button-is-pressed-in-python-tkinter/

def smooth(a,WSZ):  #https://stackoverflow.com/questions/40443020/matlabs-smooth-implementation-n-point-moving-average-in-numpy-python
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ    
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))

def fit_that_curve(fp_frames, _410, _470):
    #CALCULATIONS FOR BIEXPONENTIAL FIT
    popt, pcov=curve_fit(exp2,fp_frames,_410, maxfev=500000,p0=[0.01, 1e-7, 1e-5, 1e-9] )
    fit1=exp2(fp_frames,*popt)

    #LINEARLY SCALE FIT TO 470 DATA USING ROBUST FIT
    A=np.vstack([fit1,np.ones(len(fit1))]).T #https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
    slope=np.linalg.lstsq(A, _470, rcond=None)[0] #https://www.mathworks.com/help/stats/robustfit.html
    fit2=fit1*slope[0]+slope[1]

    #FIT 410 OVER 470
    fit3_=stats.linregress(_410, _470) 
    fit3=fit3_.intercept+fit3_.slope*_410

    return fit1, fit2, fit3;


def fp_split(fp_raw,z, fp_fps):
    
    roi=[]
    #Extract column headers from CSV to give user options
    for i in fp_raw:
        if 'Region' in i:
            roi.append(i)
        else:
            continue
    #Indexing column headers, not values
    roi_=roi[z]

    led=fp_raw['LedState']
    trace=fp_raw[roi_]

    # print(trace)
    _470_=[]
    _410_=[]
    for i, j in zip(led,trace):
        if i==1:
            _470_.append(j)
        else:
            _410_.append(j)
            
    if len(_470_)!=len(_410_):
        if len(_470_)>len(_410_):
            _470_.remove(_470_[-1])
        elif len(_470_)<len(_410_):
            _410_.remove(_410_[-1])

    _470=np.array(_470_)
    _410=np.array(_410_)

    frame=np.array([i for i in range(len(_470))])

    time=np.array([i/fp_fps for i in frame])

    return frame, time, _470, _410;
##Currently not curvy enough. Giving a straight line. Play with later
def exp2(x, a, b, c, d):
    return a*exp(b*x)+ c*exp(d*x)

def main():
    global fp_fps, length_s, z, project_id, project_home

    #Extract variables from csv file
    fp_raw=pd.read_csv(fp_file)
    
    #SPLIT COLUMNS
    fp_frames, fp_time, _470, _410=fp_split(fp_raw,z, fp_fps)

    #CURVE FITTING
    fit1, fit2, fit3= fit_that_curve(fp_frames, _410, _470)

    #CORRECT FOR POSSIBLE LED STATE SWTICH  ##doesnt work if 410 is strong
    _fit3=fit3[2000:]
    _470_=_470[2000:]
    diff1=_fit3.max()-_fit3.min()
    diff2=_470_.max()-_470_.min()


    count=0

    for i in _fit3:
        if i==_fit3.max():
            count=+1
    print(count)


    if diff1>diff2 or count==1:
        _410_=_470
        _470=_410
        _410=_410_

        fit1, fit2, fit3= fit_that_curve(fp_frames, _410, _470)
    # NORMALIZE AND SMOOTH TRACE
    normalized=(_470-fit3)/fit3
    smooth_normal=smooth(normalized, 59)  #must be an odd number
    smooth_trace=smooth_normal-smooth_normal.min()

# PLOTS
    #RAW TRACE
    fig, ax=plt.subplots(5)
    ax[0].plot(fp_frames,_470)
    ax[0].set_title('Raw 470')
    ax[0].set(xlabel='Frame #', ylabel=r'$\Delta$F/F')

    #BIEXPONENTIAL FITjj
    ax[1].set_ylim([_410.min(), _410.max()])
    ax[1].plot(fp_frames, _410)
    ax[1].plot(fp_frames, fit1, 'r') 
    ax[1].set_title('410 with biexponential fit')  #rationale: https://onlinelibrary.wiley.com/doi/10.1002/mma.7396
    ax[1].set(xlabel='Frame #', ylabel=r'$\Delta$F/F')

    #LINEARLY SCALE FIT TO 470 DATA USING ROBUST FIT 
    ax[2].set_ylim([_470.min(), _470.max()])
    ax[2].plot(fp_frames, _470)
    ax[2].plot(fp_frames, fit2, 'r')
    ax[2].set_title('Scaled biexponential fit over 470')
    ax[2].set(xlabel='Frame #', ylabel=r'F mean pixels')
    
    #410 FIT TO 470
    ax[3].plot(fp_frames, _470)
    ax[3].plot(fp_frames, fit3, 'r')
    ax[3].set_title('410 fit over 470')
    ax[3].set(xlabel='Frame #', ylabel=r'F mean pixels')

    ax[4].plot(fp_frames,smooth_trace)

    plt.show()

    plt.figure()
    plt.plot(fp_frames, smooth_trace)

    plt.show()

main()