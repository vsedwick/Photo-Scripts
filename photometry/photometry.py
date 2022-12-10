#AUTHOR: VICTORIA SEDWICK
#ADAPTED FROM NEUROfpMETRICS AND ILARIA CARTA (MATLAB)

#https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
#https://stackoverflow.com/questions/11190735/python-matplotlib-superimpose-scatter-plots

fp_file="D:\Photometry-Fall2022\Pup Block\Trial 1-PE - Copy\Ball\3\3_split2022-09-21T17_45_29.CSV"
behav_file="D:\Photometry-Fall2022\Pup Block\Trial 1-PE\\1\Raw data-MeP-CRFR2photo_pupblock1-Trial     1.xlsx"

project_home="D:\Photometry-Fall2022\Pup Block\Trial 1-PE"
project_id='pup_1'

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

def peri_4pointevent(behav, smooth_trace_crop):
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
        for j in smooth_trace_crop[(i-(10*fp_fps)):(i-1)]:
            baseline.append(j)
        peri_baseline.update({f'{counter}':baseline})
        for k in smooth_trace_crop[i:(i+(20*fp_fps))]:
            event.append(k)
        peri_event.update({f'{counter}': event})
        counter+=1
    placement_s=[i/fp_fps for i in placement]

    return peri_baseline, peri_event, placement, placement_s;

def peri_event_splits(behav, smooth_trace_crop):
    placement=[]
    peri_baseline={}
    peri_event={}

    #IDENTIFIES THE BEGINNING OF EACH BEHAVIOR EVENT
    for i in range(len(behav)):
        a=i-1
        b=i-50
        c=i+5
        if behav[i]==1 and behav[a]==0 and behav[b]==0 and behav[c]==1:
            j=int((i/behav_fps)*fp_fps)
            placement.append(j)   
        elif behav[i]==0:
            continue
    #STORES THE BASELINE AND PERI-EVENT
    counter=1
    for i in placement:
        baseline=[]
        event=[]

    #RETRIEVES VALUES FROM THE SMOOTH NORMALIZED TRACE
        for j in smooth_trace_crop[(i-(10*fp_fps)):(i-1)]:
            baseline.append(j)
        peri_baseline.update({f'{counter}':baseline})
        for k in smooth_trace_crop[i:(i+(20*fp_fps))]:
            event.append(k)
        peri_event.update({f'{counter}': event})
        counter+=1
    placement_s=[i/fp_fps for i in placement]

    return peri_baseline, peri_event, placement, placement_s;

#GUI FOR BEHAVIOR SELECTION
def pick_behaviors(behavior):
    root=tkinter.Tk()
    root.title("Select which behaviors to plot and calculate peri-events")
    root.geometry('500x300')

    class CheckBox(tkinter.Checkbutton):   #https://stackoverflow.com/questions/50485891/how-to-get-and-save-checkboxes-names-into-a-list-with-tkinter-python-3-6-5
        boxes = []  # Storage for all buttons

        def __init__(self, master=None, **options):
            tkinter.Checkbutton.__init__(self, master, options)  # Subclass checkbutton to keep other mbehavds
            self.boxes.append(self)
            self.var = tkinter.BooleanVar()  # var used to store checkbox state (on/off)
            self.text = self.cget('text')  # store the text for later
            self.configure(variable=self.var)  # set the checkbox to use our var
    for i in behavior:
        c=CheckBox(root, text=i).pack()
   
    save_button = tkinter.Button(root, text="SAVE", command=root.destroy)
    save_button.pack()
    
    #EXECUTES GUI
    root.mainloop()
    score_behaviors=[]
    #SAVES SELECTIONS
    for box in CheckBox.boxes:
        if box.var.get():  # Checks if the button is ticked
            score_behaviors.append(box.text)
    return score_behaviors

def smooth(a,WSZ):  #https://stackoverflow.com/questions/40443020/matlabs-smooth-implementation-n-point-moving-average-in-numpy-python
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ    
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))

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
            _410.remove(_410[-1])

    frame=np.array([i for i in range(len(_470))])

    time=np.array([i/fp_fps for i in frame])

    _470=np.array(_470)
    _410=np.array(_410)

    return frame, time, _470, _410;
##Currently not curvy enough. Giving a straight line. Play with later
def exp2(x, a, b, c, d):
    return a*exp(b*x)+ c*exp(d*x)

def takeoff(behav_time, fp_time, behavior, behav_fps, behav_raw, behav_start, behav_end):
    
    # fp_addon_offset=behav_duration-fp_time[-1]
    # #rationale: the behavior starts before photometry; but the behavior video decides when the photometry should start

    offset=behav_time.max()-fp_time.max()
    cutit=int((offset*fp_fps)) ##Make sure it is rounding down
    # behav_cutit=int((offset*behav_fps)
    j=behav_raw[behavior[0]]
    behav_frames=np.array([i for i in range(len(j[behav_start:-behav_end]))])
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
        print(i)
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
    global behav_fps, fp_fps, length_s, z, project_id, project_home

    behav_start=start*behav_fps
    behav_end=end*behav_fps

    fp_start=start*fp_fps
    fp_end=end*fp_fps

    #Extract variables from csv file
    fp_raw=pd.read_csv(fp_file)
    
    #SPLIT COLUMNS
    fp_frames, fp_time, _470, _410=fp_split(fp_raw,z, fp_fps)

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
    smooth_normal=smooth(normalized, 59)  #must be an odd number
    smooth_trace=smooth_normal-smooth_normal.min()

# PLOTS
    #RAW TRACE
    # fig, ax=plt.subplots(5)
    # ax[0].plot(fp_frames,_470)
    # ax[0].set_title('Raw 470')
    # ax[0].set(xlabel='Frame #', ylabel=r'$\Delta$F/F')

    # #BIEXPONENTIAL FIT
    # ax[1].set_ylim([_410.min(), _410.max()])
    # ax[1].plot(fp_frames, _410)
    # ax[1].plot(fp_frames, fit1, 'r') 
    # ax[1].set_title('410 with biexponential fit')  #rationale: https://onlinelibrary.wiley.com/doi/10.1002/mma.7396
    # ax[1].set(xlabel='Frame #', ylabel=r'$\Delta$F/F')

    # #LINEARLY SCALE FIT TO 470 DATA USING ROBUST FIT 
    # ax[2].set_ylim([_470.min(), _470.max()])
    # ax[2].plot(fp_frames, _470)
    # ax[2].plot(fp_frames, fit2, 'r')
    # ax[2].set_title('Scaled biexponential fit over 470')
    # ax[2].set(xlabel='Frame #', ylabel=r'F mean pixels')
    
    # #410 FIT TO 470
    # ax[3].plot(fp_frames, _470)
    # ax[3].plot(fp_frames, fit3, 'r')
    # ax[3].set_title('410 fit over 470')
    # ax[3].set(xlabel='Frame #', ylabel=r'F mean pixels')

    # ax[4].plot(fp_frames,smooth_trace)

    # plt.show()


    #LOAD AND SPLIT COLUMNS FOR BEHAVIOR
    behav_raw=pd.read_excel(behav_file, header=[32], skiprows=[33])
    behavior, behav_frames, behav_time=behav_split(behav_raw, behav_fps)
    print(behavior)
    #GUI FOR BEHAVIOR SELECTION
    score_behaviors=pick_behaviors(behavior) #GUI
    # score_behaviors=['sniff', 'pup grooming OR qtip', 'digging', 'rearing', 'start']

    #ALIGN BEHAVIOR AND PHOTOMETRY
    behav_time, cut=takeoff(behav_time, fp_time, behavior, behav_fps, behav_raw,behav_start, behav_end)

    behav_scores={}
    #MAKE AS BOOLEAN ARRAY
    for i in score_behaviors:
        j=np.array(behav_raw[i][behav_start:-behav_end-cut], dtype=bool)
        behav_scores.update({f"{i}": j})

    #CROP PHOTOMETRY
    fp_frames=np.array([i for i in range(len(fp_frames[(fp_start-cut):-fp_end]))])
    fp_time=np.array([i/fp_fps for i in fp_frames])
    smooth_trace_crop=smooth_trace[(fp_start-cut):-fp_end]

    print(behav_time.max(),fp_time.max())
    
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

    ax[0].plot(fp_time,smooth_trace_crop)
    ax[0].label_outer()

    x_positions = np.arange(0,behav_time.max(),(100*behav_fps))
    plt.xticks(x_positions)  

    plt.show()

    mode = 0o666
    j= os.listdir(project_home)
    if "Behaviors" not in j:
        project_home=os.path.join(project_home, "Behaviors")
        os.mkdir(project_home, mode)
    project_home=os.path.join(project_home, "Behaviors")
    print(project_home)

    
    for i in score_behaviors:
        project_home=Path(project_home)
        behav=np.array(behav_scores[i]) 
        if i=='start':
            peri_baseline, peri_event, placement, placement_s=peri_4pointevent(behav, smooth_trace_crop)
            event_means=[mean(peri_event[i]) for i in peri_event]
            baseline_means=[mean(peri_baseline[i]) for i in peri_baseline]
        else:
            peri_baseline, peri_event, placement, placement_s=peri_event_splits(behav, smooth_trace_crop)
            event_means=[mean(peri_event[i]) for i in peri_event]
            baseline_means=[mean(peri_baseline[i]) for i in peri_baseline]
            means_to_compare=[mean(baseline_means), mean(event_means)]

        percent_baseline=[i/i*100 for i in baseline_means]
        percent_event=[i/j*100 for i,j in zip(event_means, baseline_means)]

        peri_baseline_matrix=np.array([peri_baseline[i] for i in peri_baseline])
        peri_event_matrix=np.array([peri_event[i] for i in peri_event])

        #PLOT SHADING OF ROIS FOR VALIDATION
            
        
        #SAVE VALUES
        j=os.listdir(project_home)
        if i not in j:
            k=os.path.join(project_home, i)
            os.mkdir(k, mode)
        store_project=os.path.join(project_home, i)

        np.savetxt(f'{store_project}/peri_baseline_matrix_{project_id}.csv', peri_baseline_matrix, delimiter=',', fmt= '%1.15f')
        np.savetxt(f'{store_project}/peri_event_matrix_{project_id}.csv', peri_event_matrix, delimiter=',', fmt='%1.15f')
        np.savetxt(f'{store_project}/event_means_{project_id}.csv', event_means, delimiter=',', fmt='%1.15f')
        np.savetxt(f'{store_project}/baseline_means_{project_id}.csv', baseline_means, delimiter=',', fmt='%1.15f')
        np.savetxt(f'{store_project}/means_to_compare_{project_id}.csv', means_to_compare, delimiter=',', fmt='%1.15f')
        np.savetxt(f'{store_project}/placement_s_{project_id}.csv', placement_s, delimiter=',', fmt='%1.15f')


if __name__=="__main__":  #avoids running main if this file is imported
    main()




    # for i in FP_data:
    #     col.append(i)
    # frames=FP_data[col[0]]
    # fp_time=FP_data[col[1]]
    # fp_time=np.array(fp_time)
    # _470=FP_data[col[2]] 
    # _410=FP_data[col[3]] 
