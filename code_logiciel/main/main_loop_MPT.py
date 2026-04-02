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
from numerical_LT import numerical_LT_TEST
from fonctions_MPT import (
   extract_matrix,
   MSDinterp, 
   driftCorrectTrack,
   plotAlphaDeff,
   asp_ratio,
   filter_tracks_by_lambda, 
   mean_MSD_filtered, 
   complex_modulusFFT, 
   complex_modulusLT,
   numerical_LT
   )
from fonction_import_tracks import import_tracks
from plot_analysis import ( 
        plot_msd, 
        plot_XYprojection, 
        plot_alpha_multiple, 
        plot_alpha_BT_multiple, 
        diffusion_plot_multiple, 
        diffusion_plot_BT_LN,
        g1_g2
        )

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

#Selecting tracks, lambda_1 > SIZE OF PIXEL (LAMBDA MIN = 0.2)
for movie in data: #equivalent to data[i]
    filtered_tracks = filter_tracks_by_lambda(movie["tracks"], lambda_min=0.22) #call of the filter function
    
    movie["D2_filtered"]           = np.array([trk["D2"]           for trk in filtered_tracks.values()])
    movie["alpha2_filtered"]       = np.array([trk["alpha2"]       for trk in filtered_tracks.values()])
    movie["aspect_ratio_filtered"] = np.array([trk["aspect_ratio"] for trk in filtered_tracks.values()])
    movie["lambda_1_filtered"]     = np.array([trk["lambda_1"]     for trk in filtered_tracks.values()])
    movie["lambda_2_filtered"]     = np.array([trk["lambda_2"]     for trk in filtered_tracks.values()])
    movie["numTracks_filtered"]    = len(filtered_tracks)
# MSD2D de chaque trajectoire filtrée → shape (N_filtered, 499)
      # MSD2D avec padding NaN pour trajectoires de longueurs différentes
    MSD_list = [trk["MSD2D"] for trk in filtered_tracks.values()]
    
    
    max_len = max(len(m) for m in MSD_list)
    MSD_padded = np.full((len(MSD_list), max_len), np.nan)
    for j, m in enumerate(MSD_list):
        MSD_padded[j, :len(m)] = m
    movie["MSD2Dmat_filtered"] = MSD_padded  # shape (N_filtered, max_len)
    #Computes the mean of all the filtered tracks 
    movie["MSD2Dmean_filtered"] = mean_MSD_filtered(movie) #en um^2 
    movie["MSD2Dmean_filtered_m2"]  = movie["MSD2Dmean_filtered"] * 1e-12  # en m²
    
    #Computation  of G_elastic and G_loss after filter 
    MSD_filteredmean = movie["MSD2Dmean_filtered_m2"] #array
    MSD_filteredmean = MSD_filteredmean[1:169]
    dt = movie["dt"]
    
    omega, G_elastic, G_visc = complex_modulusLT(MSD_filteredmean, dt=0.03, a=0.255e-6, degree=5)
    movie["omega"] = omega
    movie["G_elastic"] = G_elastic
    movie["G_viscous"] = G_visc

print('flag2')

# %% -- ANALYSIS // PLOTS -- 

#XY PROJECTION 
plot_XYprojection(data)

#MSD OVER TIME 
plot_msd(data)

#Distribution de alpha 
plot_alpha_BT_multiple(data)

#Distribution de D 
diffusion_plot_BT_LN(data)

g1_g2(data)



