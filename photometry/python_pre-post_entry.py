#AUTHOR: VICTORIA SEDWICK
#ADAPTED FROM NEUROfpMETRICS AND ILARIA CARTA (MATLAB)

#https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
#https://stackoverflow.com/questions/11190735/python-matplotlib-superimpose-scatter-plots

rootdir="D:\Photometry-Fall2022\Pup Block\Trial 1-PE - Copy\Ball"

#PROCEED WITH CAUTION CHANGING ANYTHING AFTER THIS LINE
#MAKE SURE ALL PACKAGES ARE INSTALLED
import pandas as pd
import numpy as np
import os
import csv

root=os.path.join(rootdir,"Behaviors")
states=os.listdir(root)

# start_times=list(csv.reader(open(start_path)))

entry=start_times[0]
exit=start_times[1]



#ACCESS FOLDERS OF BEHAVIORS
means_stacked=[]
for i in behaviors:
    if i=='start':
        continue
    else:
        _=os.path.join(root,i)
        k=os.listdir(_)
        event_means=[]
        baseline_means=[]
        normal_average=[]
        
        #ACCESS INDIVIDUAL FILES
        for j in k:
            if 'means_to_compare' in j:
                l=os.path.join(_,j)
                csvfile=list(csv.reader(open(l)))
                event_means.append(csvfile[1])
                baseline_means.append(csvfile[0]) #need to drop the index and details
        #NORMALIZED AVERAGE
        if len(event_means)!=0:
            for x,y in zip(event_means, baseline_means):
                x=float(x[0])
                y=float(y[0])
                a=y/y*100
                b=x/y*100
                normal_average.append([a,b])            
            np.savetxt(f'{_}/pre_percentDelF_{i}.csv', normal_average, delimiter=',', fmt='%1.15f')
            np.savetxt(f'{_}/post_percentDelF_{i}.csv', normal_average, delimiter=',', fmt='%1.15f')
            np.savetxt(f'{_}/during_percentDelF_{i}.csv', normal_average, delimiter=',', fmt='%1.15f')  #{store_project}/means_to_compare_{project_id}


    #AREA UNDER THE CURVE; RAW DATA
        
    
print(i, event_means, baseline_means)


            # means_stacked.append([m])


            # l=pd.read_csv(_1)
            # for __ in l:
            #     print(i,__)
            # means_stacked.append([l[0], l[1]])   
# df_list = []
# for f in files:
#    df_list.append({
#       "filename": f,
#       "df": pd.read_csv(f)
#    })