#AUTHOR: VICTORIA SEDWICK
#ADAPTED FROM NEUROfpMETRICS AND ILARIA CARTA (MATLAB)

#https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
#https://stackoverflow.com/questions/11190735/python-matplotlib-superimpose-scatter-plots

fp_file="D:\Photometry-Fall2022\Pup Block\Trial 1-PE - Copy\Ball\\3\\3_split2022-09-21T17_45_29.CSV"
behav_file="D:\Photometry-Fall2022\Pup Block\Trial 1-PE - Copy\Ball\\3\Raw data-MeP-CRFR2photo_pupblock1-Trial    24 (2).xlsx"

project_home="D:\Photometry-Fall2022\Pup Block\Trial 1-PE"
project_id='pup_3'

behav_fps=30
fp_fps=20
start=114
end=391

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


def peri_4pointevent(behav, smooth_trace):
    placement=[]
    peri_baseline={}
    peri_event={}

    #IDENTIFIES THE BEGINNING OF EACH BEHAVIOR EVENT
    for i in range(len(behav)):
        if behav[i]==1:
            j=int((i/behav_fps)*fp_fps)
            placement.append(j)   
        elif behav[i]==0:
            continue
    placement=[placement[0], placement[-1]]
    #STORES THE BASELINE AND PERI-EVENT
    counter=1
    for i in placement:
        baseline=[]
        event=[]

    #RETRIEVES VALUES FROM THE SMOOTH NORMALIZED TRACE
        for j in smooth_trace[(i-(10*fp_fps)):(i-1)]:
            baseline.append(j)
        peri_baseline.update({f'{counter}':baseline})
        for k in smooth_trace[i:(i+(20*fp_fps))]:
            event.append(k)
        peri_event.update({f'{counter}': event})
        counter+=1
    placement_s=[i/fp_fps for i in placement]

    return peri_baseline, peri_event, placement, placement_s;

def takeoff(behav_time, fp_time, behavior, behav_fps, behav_raw):
    
    # fp_addon_offset=behav_duration-fp_time[-1]
    # #rationale: the behavior starts before photometry; but the behavior video decides when the photometry should start

    offset=behav_time.max()-fp_time.max()
    cutit=int((offset*behav_fps)) ##Make sure it is rounding down
    # behav_cutit=int((offset*behav_fps)
    j=behav_raw[behavior[0]]
    behav_frames=np.array([i for i in range(len(j[cutit:]))])
    behav_times=[]
    for i in behav_frames:
        j=i/behav_fps
        behav_times.append(j)
    behav_times=np.array(behav_times)
    return behav_times, cutit;

def behav_split(behav_raw, behav_fps):
    #GET BEHAVIOR HEADERS
    behaviors=[]
    for i in behav_raw:
        behaviors.append(i)
    behavior=behaviors[7:-1] #only registers columns 8 to the second to last as behaviors; ignores 'Result 1'

    #FRAME NUMBERS
    behav_frames=[]
    k=0
    for i in range(len(behav_raw[behavior[0]])):
        behav_frames.append(k)
        k+=1        
    behav_frames=behav_frames
    
    #TIME
    behav_time=np.array([i/behav_fps for i in behav_frames])

    return behavior, behav_frames, behav_time;

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
    # print(count)


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

    #BIEXPONENTIAL FIT
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


    #LOAD AND SPLIT COLUMNS FOR BEHAVIOR
    behav_raw=pd.read_excel(behav_file, header=[32], skiprows=[33])
    behavior, behav_frames, behav_time=behav_split(behav_raw, behav_fps)
    # print(behavior)
    #GUI FOR BEHAVIOR SELECTION
    # score_behaviors=['start'] #GUI
    score_behaviors=['sniff', 'pup grooming OR qtip', 'digging', 'rearing', 'start']

    #ALIGN BEHAVIOR AND PHOTOMETRY
    behav_time, cut=takeoff(behav_time, fp_time, behavior, behav_fps, behav_raw)

    behav_scores={}
    #MAKE AS BOOLEAN ARRAY
    for i in score_behaviors:
        j=np.array(behav_raw[i][cut:], dtype=bool)
        behav_scores.update({f"{i}": j})

    #CROP PHOTOMETRY
    # fp_frames=np.array([i for i in range(len(fp_frames))])
    # fp_time=np.array([i/fp_fps for i in fp_frames])
    # smooth_trace_crop=smooth_trace[(fp_start-cut):-fp_end]

    # print(behav_time.max(),fp_time.max())
    
    #PLOT ETHOGRAM   https://stackoverflow.com/questions/25469950/matplot-how-to-plot-true-false-or-active-deactive-data
    data = np.array([behav_scores[i] for i in score_behaviors])

    # create some labels
    labels = [ i for i in score_behaviors]

    # # create a color map with random colors
    colmap = matplotlib.colors.ListedColormap(np.random.random((21,3)))
    colmap.colors[0] = [1,1,1]
    # # create some colorful data:
    data_color = (1 + np.arange(data.shape[0]))[:, None] * data

    extent=[behav_time.min(), behav_time.max(), 5, -5]

    #ETHOGRAM ALIGNED WITH NORMALIZED TRACE
    etho_fig = plt.figure()
    gs = etho_fig.add_gridspec(2, hspace=0)
    ax = gs.subplots(sharex=True)

    ax[1].imshow(data_color, aspect='auto', cmap=colmap, interpolation='nearest', extent=extent)
    ax[1].set_yticks(np.arange(len(labels)))
    ax[1].set_yticklabels(labels)  ##fix y labels and tighten the graph

    ax[0].plot(fp_time,smooth_trace)
    ax[0].label_outer()
    ax[0].set_ylim([smooth_trace.min()-(0.30*smooth_trace.min()), smooth_trace.max()+(0.30*smooth_trace.max())])

    x_positions = np.arange(0,behav_time.max(),(100*behav_fps))
    plt.xticks(x_positions)  

    behav=np.array(behav_scores['start'])
    
    peri_baseline, peri_event, placement, placement_s=peri_4pointevent(behav, smooth_trace)

    # ax[0].fill_between(range(len([placement[0:-1]])), placement[0], placement[-1], alpha=0.5)
    plt.axvspan(fp_time[placement_s[0]], fp_time[placement_s[1]])
    plt.show()

main()