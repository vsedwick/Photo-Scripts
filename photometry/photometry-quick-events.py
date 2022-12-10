#AUTHOR: VICTORIA SEDWICK
#ADAPTED FROM NEUROPHOTOMETRICS AND ILARIA CARTA (MATLAB)

#This code is for mass UNSUPERVISED peri-event analysis

#https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
#https://stackoverflow.com/questions/11190735/python-matplotlib-superimpose-scatter-plots

#THIS CODE IS WRITTEN WITHOUT PLOTS; JUST FOR QUICK ACQUISITION OF PERI-EVENT VALUES FOR MANY FILES

#have each photo and behavior file for each animal in individual folders
#This code will need to be adapted if you have multiple point events other than 'start'; see lines 321 and 347

project_home="D:\Photometry-Fall2022\Pup Block\Trial 1-PE - Copy\Ball"
example_behav_file="D:\MeP-CRFR2photo_pupblock1\Export Files\Raw data-MeP-CRFR2photo_pupblock1-Trial     1 (2).xlsx"

behav_fps=30
fp_fps=20

pre_s=10
post_s=20

#Which roi/fiber signal do you want to plot (0-2)?
z=0

#name of start behavior
lolo='start'

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

def count(j):
    k=0
    for i in j:
        k+=1
    return k

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

def peri_event_splits(behav, smooth_trace):
    placement=[]
    peri_baseline={}
    peri_event={}

    #IDENTIFIES THE BEGINNING OF EACH BEHAVIOR EVENT
    for i in range(len(behav)):
        a=i-1
        b=i-50
        c=i+10
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
        for j in smooth_trace[(i-(10*fp_fps)):(i-1)]:
            baseline.append(j)
        peri_baseline.update({f'{counter}':baseline})
        for k in smooth_trace[i:(i+(20*fp_fps))]:
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
            _410.remove(_410(-1))

    frame=np.array([i for i in range(len(_470))])

    time=np.array([i/fp_fps for i in frame])

    _470=np.array(_470)
    _410=np.array(_410)

    return frame, time, _470, _410;
##Currently not curvy enough. Giving a straight line. Play with later
def exp2(x, a, b, c, d):
    return a*exp(b*x)+ c*exp(d*x)

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

def save_me(peri_baseline_matrix, store_project, project_id, peri_event_matrix, event_means, baseline_means, placement_s, i):
    if i!='start':
        if count(baseline_means)!=0 and count(event_means)!=np.nan and count(baseline_means)!=np.nan:
            means_to_compare=[mean(baseline_means), mean(event_means)]
            if count(means_to_compare)!=0 or means_to_compare!=np.nan: 
                np.savetxt(f'{store_project}/means_to_compare_{project_id}.csv', means_to_compare, delimiter=',', fmt='%s')

    if count(peri_baseline_matrix)!=0:
        np.savetxt(f'{store_project}/peri_baseline_matrix_{project_id}.csv', peri_baseline_matrix, delimiter=',', fmt= '%s')
    if count(peri_event_matrix)!=0:
        np.savetxt(f'{store_project}/peri_event_matrix_{project_id}.csv', peri_event_matrix, delimiter=',', fmt='%s')
    if count(event_means)!=0:
        np.savetxt(f'{store_project}/event_means_{project_id}.csv', event_means, delimiter=',', fmt='%s')
    if count(baseline_means)!=0:
        np.savetxt(f'{store_project}/baseline_means_{project_id}.csv', baseline_means, delimiter=',', fmt='%s')
    if count(placement_s)!=0: 
        np.savetxt(f'{store_project}/placement_s_{project_id}.csv', placement_s, delimiter=',', fmt='%s')


def main():
    global behav_fps, fp_fps, project_home, example_behav_file, z

    find=pd.read_excel(example_behav_file, header=[32], skiprows=[33])
    behavior_exp, behav_frames_exp, behav_time_exp=behav_split(find, behav_fps)
    
    #GUI FOR BEHAVIOR SELECTION
    score_behaviors=pick_behaviors(behavior_exp) #GUI

    iterations=0

    root=os.listdir(project_home)
    for animal in root:
        project_id=animal
        animal_path=os.path.join(project_home, animal)
        file_to_load=os.listdir(animal_path)
        files=[]
        #LOAD FILES
        for i in file_to_load:
            if i.endswith('.csv') or i.endswith('.CSV'):
                file=rf'{animal_path}\{i}'
                for i in file:
                    if i=='\\':
                        file.replace(i,'\\\\')
                files.append(file)
                continue
            if 'Raw data' in i and i.endswith('.xlsx'):
                file=rf'{animal_path}\{i}'
                for i in file:
                    if i=='\\':
                        file.replace(i,'\\\\')
                files.append(file)
                continue
        if len(files)!=2:
            continue
        else:
        #Extract variables from csv file

            fp_raw=pd.read_csv(files[0])
            fp_frames, fp_time, _470, _410=fp_split(fp_raw,z, fp_fps)

            #CURVE FITTING
            fit1, fit2, fit3= fit_that_curve(fp_frames, _410, _470)

            #CORRECT FOR POSSIBLE LED STATE SWTICH  ##doesnt work if 410 is strong
            _fit3=fit3[2000:]
            _470_=_470[2000:]
            diff1=_fit3.max()-_fit3.min()
            diff2=_470_.max()-_470_.min()


            c=0

            for i in _fit3:
                if i==_fit3.max():
                    c=+1
            print(c)

            if diff1>diff2 or c==1:
                _410_=_470
                _470=_410
                _410=_410_

                fit1, fit2, fit3= fit_that_curve(fp_frames, _410, _470)

        #NORMALIZE AND SMOOTH TRACE
            normalized=(_470-fit3)/fit3
            smooth_normal=smooth(normalized, 59)  #must be an odd number
            smooth_trace=smooth_normal-smooth_normal.min()

        #LOAD AND SPLIT COLUMNS FOR BEHAVIOR
            behav_raw=pd.read_excel(files[1], header=[32], skiprows=[33])
            behavior, behav_frames, behav_time=behav_split(behav_raw, behav_fps)

            #ALIGN BEHAVIOR AND PHOTOMETRY
            behav_time, cut=takeoff(behav_time, fp_time, behavior, behav_fps, behav_raw)

            behav_scores={}
            #MAKE AS BOOLEAN ARRAY
            for i in score_behaviors:
                j=np.array(behav_raw[i][cut:], dtype=bool)
                behav_scores.update({f"{i}": j})

            #CROP PHOTOMETRY
            print(behav_time.max(),fp_time.max())
            
            mode = 0o666
            j= os.listdir(project_home)
            if "Behaviors" not in j:
                behavior_path=os.path.join(project_home, "Behaviors")
                os.mkdir(behavior_path, mode)
            behavior_paths=os.path.join(project_home, "Behaviors")


##START TIME %%SPLIT THE PRE-ENTRY AND POST-ENTRIES
            start_times=behav_scores[lolo]
            place=[]
            for i in range(len(start_times)):
                if start_times[i]==1:
                    j=int(i)
                    place.append(j)   
                elif start_times[i]==0:
                    continue
            place_s=[place[0], place[-1]]
            

            ##SCORES DURING STIMULUS EXPOSURE
            for i in score_behaviors:
                behavior_path=Path(behavior_paths)
                behav=np.array(behav_scores[i][place_s[0]:place_s[1]]) 
                if not behav.any()==1:
                    continue
                if i=='start':
                    peri_baseline, peri_event, placement, placement_s=peri_4pointevent(behav, smooth_trace)
                else:
                    peri_baseline, peri_event, placement, placement_s=peri_event_splits(behav, smooth_trace)
                event_means=[mean(peri_event[i]) for i in peri_event]
                baseline_means=[mean(peri_baseline[i]) for i in peri_baseline]

                percent_baseline=[i/i*100 for i in baseline_means]
                percent_event=[i/j*100 for i,j in zip(event_means, baseline_means)]

                peri_baseline_matrix=np.array([peri_baseline[i] for i in peri_baseline])
                peri_event_matrix=np.array([peri_event[i] for i in peri_event])

                iterations+=1
                print(iterations)
                #SAVE VALUES
                j=os.listdir(behavior_path)
                p='during'
                if p not in j:
                    k=os.path.join(behavior_path, p)
                    os.mkdir(k, mode)
                state=os.path.join(behavior_path, p)
                list_state=os.listdir(state)
                if i not in list_state:
                    oo=os.path.join(state, i)
                    os.mkdir(oo, mode)
                store_project=os.path.join(state, i)
                
                save_me(peri_baseline_matrix, store_project, project_id, peri_event_matrix, event_means, baseline_means, placement_s, i)

            ##SCORES DURING STIMULUS EXPOSURE
            for i in score_behaviors:
                behavior_path=Path(behavior_paths)
                behav=np.array(behav_scores[i][place_s[0]:place_s[1]]) 
                if not behav.any()==1:
                    continue
                if i=='start':
                    peri_baseline, peri_event, placement, placement_s=peri_4pointevent(behav, smooth_trace)
                else:
                    peri_baseline, peri_event, placement, placement_s=peri_event_splits(behav, smooth_trace)
                event_means=[mean(peri_event[i]) for i in peri_event]
                baseline_means=[mean(peri_baseline[i]) for i in peri_baseline]

                percent_baseline=[i/i*100 for i in baseline_means]
                percent_event=[i/j*100 for i,j in zip(event_means, baseline_means)]

                peri_baseline_matrix=np.array([peri_baseline[i] for i in peri_baseline])
                peri_event_matrix=np.array([peri_event[i] for i in peri_event])

                    
                #SAVE VALUES
                j=os.listdir(behavior_path)
                p='before entry'
                if p not in j:
                    k=os.path.join(behavior_path, p)
                    os.mkdir(k, mode)
                state=os.path.join(behavior_path, p)
                state_list=os.listdir(state)
                if i not in state_list:
                    k=os.path.join(state, i)
                    os.mkdir(k, mode)
                store_project=os.path.join(state, i)
        
                save_me(peri_baseline_matrix, store_project, project_id, peri_event_matrix, event_means, baseline_means, placement_s, i)

            # ##SCORES DURING STIMULUS EXPOSURE
            # for i in score_behaviors:
            #     behavior_path=Path(behavior_paths)
            #     behav=np.array(behav_scores[i][place_s[1]:]) 
            #     if not behav.any()==1:
            #         continue
            #     else:
            #         peri_baseline, peri_event, placement, placement_s=peri_4pointevent(behav, smooth_trace)
            #         event_means=[mean(peri_event[i]) for i in peri_event]
            #         baseline_means=[mean(peri_baseline[i]) for i in peri_baseline]

            #         percent_baseline=[i/i*100 for i in baseline_means]
            #         percent_event=[i/j*100 for i,j in zip(event_means, baseline_means)]

            #         peri_baseline_matrix=np.array([peri_baseline[i] for i in peri_baseline])
            #         peri_event_matrix=np.array([peri_event[i] for i in peri_event])

                    
            #         #SAVE VALUES
            #         j=os.listdir(behavior_path)
            #         p='after exit'
            #         if p not in j:
            #             k=os.path.join(behavior_path, p)
            #             os.mkdir(k, mode)
            #         state=os.path.join(behavior_path, p)
            #         state_list=os.listdir(state)
            #         if i not in state_list:
            #             k=os.path.join(state, i)
            #             os.mkdir(k, mode)
            #         store_project=os.path.join(state, i)

            #         save_me(peri_baseline_matrix, store_project, project_id, peri_event_matrix, event_means, baseline_means, placement_s, i)

        project_home=Path(project_home)

if __name__=="__main__":  #avoids running main if this file is imported
    main()




#         # for i in FP_data:
#         #     col.append(i)
#         # frames=FP_data[col[0]]
#         # fp_time=FP_data[col[1]]
#         # fp_time=np.array(fp_time)
#         # _470=FP_data[col[2]] 
#         # _410=FP_data[col[3]] 
