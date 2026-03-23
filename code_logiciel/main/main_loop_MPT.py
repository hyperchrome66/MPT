#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 15:23:27 2026

@author: xuan_nguyen
"""

import numpy as np;
import os
import tkinter as tk
from tkinter import filedialog
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
#from scipy.interpolate import interp1d  # Import interp1d
from fonctions_MPT import (
   extract_matrix,
   MSDinterp, 
   driftCorrectTrack,
   plotAlphaDeff,
   asp_ratio,
   )
from fonction_import_tracks import import_tracks
from plot_analysis import plot_msd, plot_XYprojection, plot_alpha_multiple, diffusion_plot_multiple, diffusion_plot_BT_LN

#%%


""" ########## MAIN ############# """
# global px=[0.526, 0.526,2]  # NB: the tracked coordinates x,y,z are already in micron


# Create a root window for robustness (it won't be displayed)
root = tk.Tk()
root.withdraw()  # Hide the root window
directory_path = filedialog.askdirectory() #dialogue 

# directory_path='C:/PYTHON/Analysis/Analysis202405/'
data=[]

# Iterate over all files in the directory
k=0
for filename in os.listdir(directory_path):
    # Construct full file path
    path_im = os.path.join(directory_path, filename)
    # Check if the path is a file (not a directory)
    if (os.path.isfile(path_im) & filename.endswith('csv')):
        # Call the function to analyze the file
        print(filename)
        
        #Importation des trajectoires 
        tracks=import_tracks(path_im)
        
        dt=tracks[0]["dt"]
        MSD2Dmat=extract_matrix(tracks,"MSD2D")
        
        # Définir la zone (B ou T) automatiquement
        if ("zoneB" in filename) or ("Bzone" in filename) or ("B_zone" in filename):
            zone_LN = "B"
        elif ("zoneT" in filename) or ("Tzone" in filename) or ("zone_T" in filename):
            zone_LN = "T"
        else:
            zone_LN = "Unknown"
            
       # Define the lymph node type 
        if ("ing" in filename) or ("inguinal" in filename):
           type_LN = "Inguinal"
        elif ("ax" in filename) or ("axillary" in filename) :
           type_LN = "Axillary"
        elif ("mesent" in filename) or ("mesen" in filename) : 
           type_LN = "Mesenteric"
        elif ("mand" in filename) or ("mandibulaire" in filename):
           type_LN = "Mandibular"
        else : 
            type_LN = "Unknown"
            
        #Extraction du numéro du LN 
        import re 
        match = re.search(r"LN\d+", filename)
        LN = match.group() if match else "LN0"
        
        #Precise if it is stimulated or not 
        if ("CTRL" in filename) : 
            condition = "Control"
        elif ("STIM" in filename) : 
            condition = "Stimulated"
        else : 
            condition = "Unknown"
            
        
        data.append({
            "name": filename,
            "zone" : zone_LN,
            "LN" : LN, 
            "type" : type_LN, 
            "condition" : condition,
            "tracks": tracks,
            "dt": dt,
            "numTracks": len(tracks),
            "MSD2Dmat": MSD2Dmat  # be ready to add a string field like axillary, inguinal, mensenteric, female
            }) 
        k=k+1
print("flag1")

for i, movie in enumerate(data):
   # movie["tracks"], xdrift, ydrift, zdrift=driftCorrectTrack(data,i, 0);
    tracks=movie["tracks"];
   # movie["xdrift"] = xdrift;
   # movie["ydrift"] = ydrift;
   # movie["zdrift"] = zdrift;
   # movie["XCcorr"]= np.array([trk["XCcorr"] for trk in tracks.values()]);
   # movie["YCcorr"]= np.array([trk["YCcorr"] for trk in tracks.values()]);
   # movie["ZCcorr"]= np.array([trk["ZCcorr"] for trk in tracks.values()]);
    movie["MSDmat"] = extract_matrix(tracks,"MSD2D");
   #movie["MSDcmat"]= extract_matrix(tracks,"MSDcorr");
    movie["D2"]= np.array([trk["D2"] for trk in tracks.values()]);
    movie["alpha2"]= np.array([trk["alpha2"] for trk in tracks.values()]);
    movie["aspect_ratio"] = np.array([trk["aspect_ratio"] for trk in tracks.values()]);
    movie["lambda_1"] = np.array([trk["lambda_1"] for trk in tracks.values()]);
    movie["lambda_2"] = np.array([trk["lambda_2"] for trk in tracks.values()]);

data=MSDinterp(data)    
print('flag2')


# %% -- ANALYSIS // PLOTS -- 

#XY PROJECTION 
plot_XYprojection(data)

#MSD OVER TIME 
plot_msd(data)

#Distribution de alpha 
plot_alpha_multiple(data)

#Distribution de D 
diffusion_plot_BT_LN(data)



