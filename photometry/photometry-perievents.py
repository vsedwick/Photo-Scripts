
#AUTHOR:
#ADAPTED FROM NEUROfpMETRICS AND ILARIA CARTA (MATLAB)

#https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
#https://stackoverflow.com/questions/11190735/python-matplotlib-superimpose-scatter-plots

##Stopped at smoothing


#FIX FRAME RATE OF fp TO MATCH FRAME RATE OF VIDEO

start=60
length_s=300
behav_fps=30
fp_fps=20
#Which roi/fiber signal do you want to plot (0-2)
z=0
fp_file="D:\Photometry-Fall2022\Pup Block\Trial 1-PE\\7\PUP\\7_split2022-09-20T21_12_14.CSV"
behav_file="D:\MeP-CRFR2photo_pupblock1\Export Files\Raw data-MeP-CRFR2photo_pupblock1-Trial    11 (2).xlsx"

save_events_locations="V:\\"

_id=3

import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numpy import exp
import numpy as np
from numpy import mean
from scipy import stats
import tkinter
# from Tkinter import * #https://www.geeksforgeeks.org/how-to-create-a-pop-up-message-when-a-button-is-pressed-in-python-tkinter/
# import Tkinter.messagebox
# import tkinter as tk

def peri_event_splits(behav, smooth_trace):
    placement=[]
    peri_baseline={}
    peri_event={}

    #IDENTIFIES THE BEGINNING OF EACH BEHAVIOR EVENT
    for i in range(len(behav)):
        a=i-1
        b=i-50
        c=i+5
        if behav[i]==1 and behav[a]==0 and behav[b]==0 and behav[c]==1:
            j=int((i*behav_fps)/fp_fps)
            placement.append(j)   
        elif behav[i]==0:
            continue

    #STORES THE BASELINE AND PERI-EVENT
    counter=1
    for i in placement:
        baseline=[]
        event=[]

    #RETRIEVES VALUES FROM THE SMOOTH NORMALIZED TRACE
        for j in smooth_trace[(i-(10*fp_fps)):(i-1)]:
            baseline.append(j)
        peri_baseline.update({f"{counter}":baseline})
        for k in smooth_trace[i:(i+(20*fp_fps))]:
            event.append(k)
        peri_event.update({f"{counter}":event})
        counter+=1

    return peri_baseline, peri_event, placement;

def smooth(a,WSZ):  #https://stackoverflow.com/questions/40443020/matlabs-smooth-implementation-n-point-moving-average-in-numpy-python
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ    
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))

def split(fp_raw,z, fp_fps):
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
    _470=[]
    _410=[]
    for i, j in zip(led,trace):
        if i==6:
            _470.append(j)
        else:
            _410.append(j)

    if len(_470)!=len(_410):
        if len(_470)>len(_410):
            _470.remove(_470[-1])
        elif len(_470)<len(_410):
            _410.remove(_410(-1))

    k=0
    frame=[]
    for i in _470:
        frame.append(k)
        k+=1

    time=[]
    for i in frame:
        j=i/fp_fps
        time.append(j)

    FP_data=pd.DataFrame({
            'Frames': frame,
            'Time': time, 
            '470': _470,
             '410': _410,
             })

    return FP_data
##Currently not curvy enough. Giving a straight line. Play with later
def exp2(x, a, b, c, d):
    return a*exp(b*x)+ c*exp(d*x)

def takeoff(behav_duration, fp_time, behavior, behav_fps, behav_raw):
    offset=behav_duration-fp_time[-1]
    cutit=int((offset*behav_fps).round())  ##Make sure it is rounding down
    behav_frames=[]
    k=0
    for i in range(len(behav_raw[behavior[0]][cutit:])):
        behav_frames.append(k)
        k+=1      
    behav_frames=np.array(behav_frames)
    behav_time=[]
    for i in behav_frames:
        j=i/behav_fps
        behav_time.append(j)
    behav_time=np.array(behav_time)
    return behav_frames, behav_time, cutit

def main():
    global behav_fps, fp_fps, length_s, start, z

    #Extract variables from csv file
    fp_raw=pd.read_csv(fp_file)
    
    #Split columns and channels
    col=[]
    FP_data=split(fp_raw,z, fp_fps)
    for i in FP_data:
        col.append(i)
    fp_frames=np.array(FP_data[col[0]])
    fp_time=np.array(FP_data[col[1]])
    _470=FP_data[col[2]] 
    _410=FP_data[col[3]] 

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

    #NORMALIZE AND SMOOTH TRACE
    normalized=(_470-fit3)/fit3
    smooth_normal=smooth(normalized, 71)
    smooth_trace=smooth_normal-smooth_normal.min()


    behav_raw=pd.read_excel(behav_file, header=[32], skiprows=[33])

    #GET BEHAVIOR HEADERS
    behaviors=[]
    for i in behav_raw:
        behaviors.append(i)
    behavior=behaviors[7:-1] #only registers columns 8 to the second to last as behaviors; ignores 'Result 1'
    
    #ALIGN BEHAVIOR AND PHOTOMETRY
    behav_frame_duration=len(behav_raw[behavior[0]])
    behav_duration=behav_frame_duration/behav_fps
    behav_frames, behav_time, cut=takeoff(behav_duration, fp_time, behavior, behav_fps, behav_raw)
    behav_raw_cut=behav_raw[cut:]

    # SAVE EXCEL FILE FOR EACH BASELINE
    behav=np.array(behav_raw_cut['sniff'][cut:]) 
    #PERI-EVENTS
    peri_baseline, peri_event, placement=peri_event_splits(behav, smooth_trace)  #all events should automatically be the same size
    event_means=[]
    baseline_means=[]
    for i in peri_baseline:
        j=mean(peri_baseline[i])
        baseline_means.append(j)
    for i in peri_event:
        j=mean(peri_event[i])
        event_means.append(j)
    print(event_means)  #save event means; peri_baseline; peri_event, baseline means

    
    # print(baseline_means)

    # peri_baseline={}
    # peri_event={}
    # behav=np.array(behav_raw_cut['sniff'][cut:])
    # # print(behav)
    # for i in range(len(behav)):
    #     a=i-1
    #     b=i-50
    #     c=i+5
    #     if behav[i]==1 and behav[a]==0 and behav[b]==0 and behav[c]==1:
    #         # print(i)
    #         placement=[]
    #         baseline=[]
    #         event=[]
    #         counter=1
    #         for j in range(len(behav_time[(i-(6*behav_fps)):(i-1)])):
    #             k=int(((j+i)*behav_fps)/fp_fps)
    #             baseline.append(smooth_trace[k])
    #         peri_baseline.update({f"{counter}":baseline})
    #         for l in range(len(behav_time[i:i+20])):
    #             m=int(((l+i)*behav_fps)/fp_fps)
    #             event.append(smooth_trace[m])
    #         peri_event.update({f"{counter}":event})
    #         counter+=1
            
    #     elif behav[i]==0:
    #         continue
        # elif behav[i]==0:
        #     continue
    # eents_stacked_matrix
    # peri_events)stacked_matrix%b
    # z_perievents
    # means to compare events %a  
    # event means%%a 
    # z events
    # pervent change
    # percent change average
    # percent change individual
    # average events%%a

    # perievents%%b
    # save_me=pd.DataFrame({
    #     f'peri-baseline': base,
    #     f'peri-event': event, 
        
    #     })
    # summary=os.path.join(save_events_locations, f'{behavior[_]}_{id_}.xlsx')
    # video_summary.to_excel(summary, index=True, header=True)

            
    #behavior bool
    #co


# fps=20;
# baseline = smooth_normalizedtrace(round(roi_s(1,1)):round(roi_s(1,1))+15*fps);
# baseline_events = cell(length(roi_s),1);                                              %preallocate cell for next loop
# events = cell(length(roi_s),1);
# peri_events = cell(length(roi_s),1);


    
    

#     data_offset = video_duration_s - recording_duration_s;
# d=recording_duration_s-data_offset;
# behav_cut=behav_scoring(data_offset*frame_rate:end,:);



# recording_duration_s = floor(length(normalizedtrace)/20); 
# data470_cut = normalizedtrace(1:recording_duration_s*20); %cropping 
# first_bin= 1:length(normalizedtrace);

# plot(smooth_normalizedtrace)




    #410_fit2=

    #ROBUST FIT  %%figure out the python equivalent for robustfit()
    
    #Df/f 410 OVER 470
    # _410to470=(_470-lin_fit)/lin_fit
    # _470_scaled=fitlm(_410, _470)   
    # normalizedtrace=(_470-_410to470)

    # fig, ax=plt.subplots(4)
    # ax[1].plot(_470)
    # ax[1].plot(_410_fit2)

    # ax[2],plot(_410to470)

    # ax[3], plot(_470_scaled)



    # plt.plot(FP_data[col[0]], exp2(FP_data[col[0]], *popt), 'r-')
# plot(FP.data(:,3))
# temp_x = 1:length(FP.data); %note -- using a temporary x variable due to computational constraints in matlab
# temp_x = temp_x';
# FP.fit1 = fit(temp_x,FP.data(:,3),'exp2');

# # hold on
# # plot(FP.fit1(temp_x),'LineWidth',2)

# # title('415 data fit with biexponential')
# # xlabel('frame number')
# # ylabel('F (mean pixel value')
# # plt.figure(1)
# # a = plt.subplot(211)
# # r = 2**16/2
# # a.set_ylim([-r, r])
# a.set_xlabel('time [s]')
# a.set_ylabel('sample value [-]')
# x = np.arange(44100)/44100
# plt.plot(x, left)
# b = plt.subplot(212)
# b.set_xscale('log')
# b.set_xlabel('frequency [Hz]')
# b.set_ylabel('|amplitude|')
# plt.plot(lf)
# plt.savefig('sample-graph.png')

#RAW TRACE


#     temp_x=createList(1,len(_))
#     FP=[]
# #linearly scale fit to 470 data using robustfit

# FP_fit2 = robustfit(FP.fit1(temp_x),FP.data(:,2)); %scale using robust fit
# FP.lin_fit = FP.fit1(temp_x)*FP.fit2(2)+FP.fit2(1); %plot with robust fit


    
    # plt.show()


    
    # for j,i in zip(trace,led):
    #     if i==6:
    #         _470.append(j)
    #     if i==1:
    #         _410.append(j)
    # print(len(_470),len(_410))
    
    
if __name__=="__main__":  #avoids running main if this file is imported
    main()




