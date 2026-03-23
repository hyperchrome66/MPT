#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 13:49:40 2024

NB: data is a list while tracks (contained in data) is a dict!

@author: paolo
"""


# import track
import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
import os
import tkinter as tk
from tkinter import filedialog
from scipy import stats
import pickle
from pathlib import Path
from scipy.interpolate import interp1d  # Import interp1d

        
def import_tracks(name):
    df=pd.read_csv(name,skiprows=[1,2,3], encoding='ISO-8859-1');
    tr=np.unique(df.TRACK_ID);
    allx=df.POSITION_X.to_numpy();
    ally=df.POSITION_Y.to_numpy();
    allz=df.POSITION_Z.to_numpy();
    allt=df.POSITION_T.to_numpy();
    allr=df.RADIUS.to_numpy();
    allq=df.QUALITY.to_numpy();
    allf=df.FRAME.to_numpy();
    
    k=0;
    tracks={};
    for i in tr:
        x=allx[np.where((df.TRACK_ID)==i)];
        y=ally[np.where((df.TRACK_ID)==i)];
        z=allz[np.where((df.TRACK_ID)==i)];
        t=allt[np.where((df.TRACK_ID)==i)];
        r=allr[np.where((df.TRACK_ID)==i)];
        q=allq[np.where((df.TRACK_ID)==i)];
        frame0=allf[np.where((df.TRACK_ID)==i)] # count from 0!!!
        a=np.argsort(t) # check order!!!
        t=t[a]
        dt=np.diff(t[0:2]).item()
        frame=np.arange(min(frame0),max(frame0)+1,1)
        t= frame*dt
        x= correctGap(x[a], frame0, frame)
        y= correctGap(y[a], frame0, frame)
        z= correctGap(z[a], frame0, frame)
        r= correctGap(r[a], frame0, frame)
        q= correctGap(q[a], frame0, frame)
        
        xc=np.nanmean(x);
        yc=np.nanmean(y);
        zc=np.nanmean(z);
        rmean=np.nanmean(r)
        qmean=np.nanmean(q)
        ngap= sum(np.isnan(x))
        msd2D=MSD2(x,y);
        msd3D=MSD3(x,y,z);
        alpha2=alphaCoeff(msd2D) 
        alpha3=alphaCoeff(msd3D)        
        slope, err2=diffCoeff(dt, msd2D) 
        D2=slope/4
        slope, err3=diffCoeff(dt, msd3D)       
        D3=slope/6
        gyr = gyradius(x,y,z)
        alphaTime, DeffTime = MSDtime(msd3D, dt, 10)  # sliding window fit alpha dn Deff (3D only)

        if (len(x)>=5):
            tracks[k]={
                "id": i, 
                "x": x, "y": y,  "z": z, "t": t, "frame": frame, "radius":r, "quality":q,
                "dt":dt, "nspots": len(x),"xc": xc, "yc": yc,  "zc": zc, "rmean": rmean,
                "qmean": qmean, "ngap": ngap,
                "MSD2D": msd2D, "alpha2": alpha2, "MSD3D": msd3D, "alpha3": alpha3,
                "D2": D2, "D3": D3, "err2": err2, "err3": err3, "gyradius": gyr, 
                "alphaTime": alphaTime, "DeffTime": DeffTime
                }
            k=k+1;
           # print(k)
    return tracks

def MSD2(x,y):
    x=np.array(x);
    y=np.array(y);
    m=np.full(len(x)-1,np.nan);
    if len(x)>5:
        for i in range(len(x)-1):
            end=len(x);
            # experimental MSD:
            m[i]=np.nanmean(np.square(x[i:end]-x[0:end-i])+np.square(y[i:end]-y[0:end-i]));
    return m;

def MSD3(x,y,z):
    x=np.array(x);
    y=np.array(y);
    z=np.array(z);
    m=np.full(len(x)-1,np.nan);
    if len(x)>5:
        for i in range(len(x)-1):
            end=len(x);
            # experimental MSD:
            m[i]=np.nanmean(np.square(x[i:end]-x[0:end-i])+np.square(y[i:end]-y[0:end-i]+np.square(z[i:end]-z[0:end-i])));
    return m;

# extract matrix from a dictionary like structure (tracks)
def extract_matrix(trk,field_name):
    max_len = max(len(entry[field_name]) for entry in trk.values())
    v_matrix = np.array([np.pad(entry[field_name], (0, max_len - len(entry[field_name])), constant_values=np.nan) for entry in trk.values()])
    return v_matrix

def diffCoeff(dt,msd3D):
    if (~np.isnan(msd3D[0])):
        t=np.arange(5)*dt
        slope, intercept = np.polyfit(t, msd3D[0:5], 1)
        return slope, intercept
    else:
        return 0, 0


def gyradius(x,y,z):
    gyr=np.sqrt(np.var(x)+np.var(y)+np.var(z))
    return gyr


def alphaCoeff(msd3D):
    if (~np.isnan(msd3D[0])):
        x=np.array(list(range(1,len(msd3D))))
        x = np.where(x <= 0, 1e-10, x)  # Replace 0 or negative values with a small number
        msd3D[1:] = np.where(msd3D[1:] <= 0, 1e-10, msd3D[1:])
        slope, intercept = np.polyfit(np.log(x), np.log(msd3D[1:]), 1)
        return slope
    else: 
        return 0

# concatenate row_wise matrices with different column length belonging to a dictionary
def concatenateMatrix(data,matrix):
    max_cols = max((entry[matrix].shape[1]) for entry in data.values())
    padded_matrices = [np.pad(entry[matrix], ((0, 0), (0, max_cols - entry[matrix].shape[1])), constant_values=np.nan) for entry in data.values()]
    result = np.vstack(padded_matrices)
    return result

def correctGap(x,frame0, frame):
    # Initialize a NaN-filled array of the same length as full_range
    filled = np.full_like(frame, np.nan, dtype=np.float64)
    filled[(np.array(frame0) - frame0[0]).astype(int)]=x
    filled=np.array(filled)
    return filled
        
 
    
"""
Correct the drift by computing the center of mass of the particles that are 
present in the movie from the beginning to the end
"""
def driftCorrectTrack(data,selMovie, doPlot):
    tracks=data[selMovie]["tracks"]
    tracksLen=np.array([tracks['nspots'] for tracks in tracks.values()])
    tracksChk=np.array([np.mean(tracks['x']) for tracks in tracks.values()])
    tracksSel=np.intersect1d(np.where(tracksLen==max(tracksLen))[0],np.where(~np.isnan(tracksChk))[0])
    
    mx = np.vstack([tracks[key]["x"] for key in tracksSel])
    xdrift=np.mean(mx,0)
    my = np.vstack([tracks[key]["y"] for key in tracksSel])
    ydrift=np.mean(my,0)
    mz = np.vstack([tracks[key]["z"] for key in tracksSel])
    zdrift=np.mean(mz,0)
    
    
   
    
    for key in tracks.keys():
       
        # Store drift values directly in the dictionary
        tracks[key]["xdrift"] = xdrift
        tracks[key]["ydrift"] = ydrift
        tracks[key]["zdrift"] = zdrift
        
        # Corrected coordinates based on drift
        time = tracks[key]["frame"]
        
        # it happens rarely but the time could be longer than the xdrift vector if particle appears during the film, this has to be trimmed
        
        time=time[time<len(xdrift)]
        
        tracks[key]["xcorr"] = tracks[key]["x"][:len(time)] - xdrift[time]+xdrift[0]
        tracks[key]["ycorr"] = tracks[key]["y"][:len(time)] - ydrift[time]+ydrift[0]
        tracks[key]["zcorr"] = tracks[key]["z"][:len(time)]- zdrift[time]+zdrift[0]
        
        tracks[key]["XCcorr"] = np.nanmean(tracks[key]["xcorr"])
        tracks[key]["YCcorr"] = np.nanmean(tracks[key]["xcorr"])
        tracks[key]["ZCcorr"] = np.nanmean(tracks[key]["zcorr"])
        
        
        # Calculate MSD (Mean Squared Displacement)
        tracks[key]["MSD3corr"] = MSD3(tracks[key]["xcorr"], tracks[key]["ycorr"], tracks[key]["zcorr"])
        tracks[key]["MSD2corr"] = MSD2(tracks[key]["xcorr"], tracks[key]["ycorr"])
        
        # Calculate alpha coefficients
        tracks[key]["alpha2corr"] = alphaCoeff(tracks[key]["MSD2corr"])
        tracks[key]["alpha3corr"] = alphaCoeff(tracks[key]["MSD3corr"])
        
        # Calculate diffusion coefficients
        slope2, _ = diffCoeff(data[selMovie]["dt"], tracks[key]["MSD2corr"])
        tracks[key]["D2corr"] = slope2 / 4
        slope3, _ = diffCoeff(data[selMovie]["dt"], tracks[key]["MSD3corr"])
        tracks[key]["D3corr"] = slope3 / 4
        print(key)

    if doPlot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Plot the 3D line
        ax.plot(xdrift, ydrift, zdrift, label='3D line')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('data '+str(selMovie))
    # Return the modified tracks dictionary
    
    return(tracks, xdrift, ydrift, zdrift)


def MSDinterp(data):
    for movie in data:
        msdMean=np.nanmean(movie["MSD3Dcmat"],axis=0)
        tracks = movie["tracks"]
        timeMax = len(msdMean)*movie["dt"]
        timeOri = np.arange(0, timeMax, movie["dt"])   # time original
        timeInt = np.arange(0, max(timeOri), 10) # time where I want to sample
        interpolator = interp1d(timeOri, msdMean, kind='quadratic')  # Linear interpolation function
        MSD3MeanInt = interpolator(timeInt) # MSD interpolated
        movie["timeOri"]=timeOri
        movie["MSD3MeanOri"]=msdMean
        movie["timeInt"]=timeInt
        movie["MSD3MeanInt"]=MSD3MeanInt
    return (data)

def selectData(data, sel):
    for i, movie in enumerate(data):
        tracks=movie['tracks']
        selection = [k for k in tracks.keys() if tracks[k]['zone']==sel]


#alphaTime, DeffTime = MSDtime(msd3D, 10) 
# the Deff is 
def MSDtime(msd3D, dt, step=5):
    if (len(msd3D)<(step+1)):
        return np.array([]),np.array([])
    t = np.arange(1, len(msd3D))*dt 
    #t=np.array(list(range(1,len(msd3D))))
    #t = np.where(t <= 0, 1e-10, t)  # Replace 0 or negative values with a small number
    msd3D[1:] = np.where(msd3D[1:] <= 0, 1e-10, msd3D[1:]) # correct MSD=<0 by small number
    Deff=np.zeros(len(msd3D)-step-1)
    alpha=np.zeros(len(msd3D)-step-1)
    for i in range (0, len(msd3D)-step-1):
        slope, intercept = np.polyfit(np.log(t[i:i+step]), np.log(msd3D[i:i+step]), 1)
        Deff[i] = np.exp(intercept)
        alpha[i] = slope
    return alpha, Deff

# this increases the window
def MSDtime2(msd3D, dt, step=5):
    if (len(msd3D)<(step+1)):
        return np.array([]),np.array([])
    t = np.arange(1, len(msd3D))*dt 
    #t=np.array(list(range(1,len(msd3D))))
    #t = np.where(t <= 0, 1e-10, t)  # Replace 0 or negative values with a small number
    msd3D[1:] = np.where(msd3D[1:] <= 0, 1e-10, msd3D[1:]) # correct MSD=<0 by small number
    Deff=np.zeros(len(msd3D)-step-1)
    alpha=np.zeros(len(msd3D)-step-1)
    for i in range (0, len(msd3D)-step-1):
        slope, intercept = np.polyfit(np.log(t[1:i+step]), np.log(msd3D[1:i+step]), 1)
        Deff[i] = np.exp(intercept)
        alpha[i] = slope
    return alpha, Deff


def plotAlphaDeff(alpha, Deff,step=5):
    N_msd = len(msd3D) # Lunghezza totale dell'array MSD (es. 37)
    N_output = len(alpha)   # Lunghezza degli array alpha e Deff (es. 37 - 5 = 32)
    T = np.arange(1, len(alpha)+ 1) 
    # --- 2. Creazione del Plot ---
    plt.figure(figsize=(10, 6))
    # Scatter plot con 'c' (colore) mappato all'array del tempo T
    scatter = plt.scatter(
        alpha, # x-axis: alpha (slope)
        Deff, # y-axis: Deff
        c=T, # Color code: Time (T)
        cmap='viridis', # Mappa dei colori (puoi usare 'jet', 'plasma', ecc.)
        s=50, # Dimensione dei punti
        alpha=0.8 # transparency (not the alpha coefficient!)
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label('Time of the window (t)')
    # Etichette e Titolo
    plt.xlabel(r'Anomalous exponent ($\alpha$)', fontsize=14)
    plt.ylabel(r'$D_{eff}$', fontsize=14)
    plt.title(rf'Time dependendence of $\alpha$ and $D_{{eff}}$ (window: {step} timepoints)', fontsize=16)

    # Linee di riferimento per la Diffusione Normale
    plt.axvline(x=1.0, color='r', linestyle='--', linewidth=1, label=r'$\alpha = 1$ (Classic diffusion)')
    # Griglia
    plt.grid(True, linestyle=':', alpha=0.6)
    # Legenda
    plt.legend()
    plt.show()
    
   
""" ########## MAIN ############# """
# global px=[0.526, 0.526,2]  # NB: the trakcked coordinates x,y,z are already in micron


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
        tracks=import_tracks(path_im)
        dt=tracks[0]["dt"]
        MSD2Dmat=extract_matrix(tracks,"MSD2D")
        MSD3Dmat=extract_matrix(tracks,"MSD3D")
        
        data.append({
            "name": filename,
            "tracks": tracks,
            "dt": dt,
            "numTracks": len(tracks),
            "MSD2Dmat": MSD2Dmat,     
            "MSD3Dmat": MSD3Dmat,        
            "Condition": "TB"  # be ready to add a string field like ascillary, inguinal, mensenteric, female
            }) 
        k=k+1
print("done")
for i, movie in enumerate(data):
    movie["tracks"], xdrift, ydrift, zdrift=driftCorrectTrack(data,i, 0);
    tracks=movie["tracks"];
    movie["xdrift"] = xdrift;
    movie["ydrift"] = ydrift;
    movie["zdrift"] = zdrift;
    movie["XCcorr"]= np.array([trk["XCcorr"] for trk in tracks.values()]);
    movie["YCcorr"]= np.array([trk["YCcorr"] for trk in tracks.values()]);
    movie["ZCcorr"]= np.array([trk["ZCcorr"] for trk in tracks.values()]);
    movie["MSD2Dcmat"] = extract_matrix(tracks,"MSD2corr");
    movie["MSD3Dcmat"]= extract_matrix(tracks,"MSD3corr");
    movie["D3corr"]= np.array([trk["D3corr"] for trk in tracks.values()]);
    movie["alpha3corr"]= np.array([trk["alpha3corr"] for trk in tracks.values()]);

data=MSDinterp(data)    

# save dictionary
import pickle
from pathlib import Path    
path = Path(directory_path)
cond=path.parent.name
with open(directory_path+'/'+cond+'.pkl', 'wb') as file:
    # Serialize and save the dictionary to the file
    pickle.dump(data, file)
    
    
    
    
    
    
   
    
