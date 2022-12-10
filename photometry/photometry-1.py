#https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
#https://stackoverflow.com/questions/11190735/python-matplotlib-superimpose-scatter-plots


start=60
length_s=300
fps_behav=30
fps_trace=20
#Which roi/fiber signal do you want to plot (0-2)
z=0


import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import warnings
    

def split(fp_raw,z):
    led=fp_raw['LedState']

    roi=[]
    #Extract column headers from CSV to give user options
    for i in fp_raw:
        if 'Region' in i:
            roi.append(i)
        else:
            continue
    trace=roi[z]
    trace=fp_raw[trace]
    # print(trace)
    _470=[]
    _410=[]
    for i, j in zip(led,trace[0:-1]):
        if (i==6):
            _470.append(j)
        else:
            _410.append(j)
    k=0
    frame=[]
    for i in _470:
        frame.append(k)
        k+=1
    print(len(frame), len(_470), len(_410))
    FP_data=pd.DataFrame({
            'Frames': frame, 
            '470': _470,
             '410': _410,
             })

    # #Ask which ROI to plot
    # while True:
    #     try:
    #         print(roi)
    #         z=int(input("Which ROI do you wish to plot? (input 0-2): "))
    #     except ValueError() or z not in roi[z]:
    #         continue
    #     else:
    #         break
    

    return FP_data

def exp2(x, a, b, c, d):
    return a*np.exp(b*x)+ c*np.exp(d*x)

def main():
    global fps_behav, fps_trace, length_s, start, z
    # global fps_trace
    # global length_s
    # global start

    #Extract variables from csv file
    fp_raw=pd.read_csv("D:\Pup Block 1\Day1-09202022\8\Pup\\7_split2022-09-20T22_27_56.CSV")
    
    #Split columns and channels
    col=[]
    FP_data=split(fp_raw,z)
    for i in FP_data:
        col.append(i)

    #For biexponential fit

    fig, ax=plt.subplots(1)
    ax[0].plot(FP_data[col[0]], FP_data[col[1]])
    ax[0].set_title('Raw 470')
    ax[0].set(xlabel='Frame #', ylabel=r'$\Delta$F/F')
    
    ax[1].plot
    # for graph in ax.flat:
        


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
    
    
main()




    