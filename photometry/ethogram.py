#https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
#https://stackoverflow.com/questions/11190735/python-matplotlib-superimpose-scatter-plots


#TO FIX LATER: ENSURE CHANNELS ARE ALL SAME LENGTH
#FIX FRAME RATE OF PHOTO TO MATCH FRAME RATE OF VIDEO

start=60
length_s=300
fps_behav=30
fps_photo=20
#Which roi/fiber signal do you want to plot (0-2)
z=0


#type 'pip install package_name' into terminal if these packages arent arent already installed
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numpy import exp
import numpy as np
import tkinter
from PIL import ImageTk, Image
import matplotlib as mpl

# def imshow_with_colorbar(image: np.ndarray,
#                          ax_handle,
#                          fig_handle: matplotlib.figure.Figure,
#                          clim: tuple = None,
#                          cmap: str = None,
#                          interpolation: str = None,
#                          symmetric: bool = False,
#                          func: str = 'imshow',
#                          **kwargs) -> matplotlib.colorbar.Colorbar:

def main():
    global fps_behav, fps_photo, length_s, start, z

    #Extract variables from csv file
    etho_raw=pd.read_excel("V:\Raw data-MeP-CRFR2photo_pupblock1-Trial    23 (2).xlsx", header=[32], skiprows=[33])
    
    #Get behavior headers
    behaviors=[]
    for i in etho_raw:
        behaviors.append(i)
    behavior=behaviors[7:-1] #only registers columns 8 to the second to last as behaviors; ignores 'Result 1'

    #FRAME NUMBERS
    etho_frames=[]
    k=0
    for i in range(len(etho_raw[behavior[0]])):
        etho_frames.append(k)
        k+=1        

    #GUI FOR BEHAVIOR SELECTION
    root=tkinter.Tk()
    root.title("Select which behaviors to plot and calculate peri-events")
    root.geometry('500x300')

    class CheckBox(tkinter.Checkbutton):   #https://stackoverflow.com/questions/50485891/how-to-get-and-save-checkboxes-names-into-a-list-with-tkinter-python-3-6-5
        boxes = []  # Storage for all buttons

        def __init__(self, master=None, **options):
            tkinter.Checkbutton.__init__(self, master, options)  # Subclass checkbutton to keep other methods
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





    fig, ax = plt.subplots()

# define the colors
    # cmap = mpl.colors.ListedColormap(['w', 'm'])

# create a normalize object the describes the limits of
# each color
    bounds = [0., 0.5, 1.]
    # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    print(score_behaviors)
# plot it

    ax.imshow(etho_frames,(etho_raw[score_behaviors[0]]==1).any(), interpolation='none', cmap='twilight')  #, norm=norm)#, suggest=True, show=False)


    # plt.imshow(etho_raw[behavior[17]], aspect='auto', interpolation='none', origin='lower')#suggest=True, show=False)
    # plt.style.use('seaborn')              
    # # ax[1].set_figheight(0.6)
    # plt.show() 
main()
    