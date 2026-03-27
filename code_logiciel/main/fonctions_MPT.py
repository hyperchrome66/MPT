#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 15:22:14 2026

@author: xuan_nguyen
"""

import numpy as np;
import matplotlib.pyplot as plt;
from scipy.interpolate import interp1d # Import interp1d
        

"""FONCTIONS PRINCIPALES
"""

#CALCUL DU MSD 
""" x and y are in um 
    m : MSD in um^2
"""
def MSD2(x,y):
    x=np.array(x);
    y=np.array(y);
    #INITIALISATION EN REMPLISSANT ARRAY AVEC NAN 
    m=np.full(len(x)-1,np.nan);
    if len(x)>5:
        for i in range(len(x)-1):
            end=len(x);
            # experimental MSD:
            m[i]=np.nanmean(np.square(x[i:end]-x[0:end-i])+np.square(y[i:end]-y[0:end-i]));
    return m;

# extract matrix from a dictionary-like structure (tracks)
def extract_matrix(trk,field_name):
    max_len = max(len(entry[field_name]) for entry in trk.values())
    v_matrix = np.array([np.pad(entry[field_name], (0, max_len - len(entry[field_name])), constant_values=np.nan) for entry in trk.values()])
    return v_matrix


#Calcule le coefficient de diffusion pour les 5 premières positions 
def diffCoeff(dt, msd2D):
    if (~np.isnan(msd2D[0])):
        t=np.arange(5)*dt #Calcul du coeff de diff pour les 5 premières positions
        slope, intercept = np.polyfit(t, msd2D[0:5], 1)
        return slope, intercept
    else:
        return 0, 0

#calcule le coeff alpha sur l'ensemble de la trajectoire 
def alphaCoeff(msd2D):
    if (~np.isnan(msd2D[0])):
        x=np.array(list(range(1,len(msd2D))))
        x = np.where(x <= 0, 1e-10, x)  # Replace 0 or negative values with a small number
        msd2D[1:] = np.where(msd2D[1:] <= 0, 1e-10, msd2D[1:])
        slope, intercept = np.polyfit(np.log(x), np.log(msd2D[1:]), 1)
        return slope
    else: 
        return 0
    
"""
#calcule le coeff alpha sur l'ensemble de la trajectoire 
def alphaCoeff(dt, msd2D):
    if (~np.isnan(msd2D[0]) & len(msd2D>10)):
        x = np.array(list(range(1,len(msd2D))))
        x = np.where(x <= 0, 1e-10, x)  # Replace 0 or negative values with a small number
        msd2D[1:] = np.where(msd2D[1:] <= 0, 1e-10, msd2D[1:])
        t = dt*x[0:len(msd2D)-6]
        y = msd2D[1:len(msd2D)-5]
        slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
        return slope 
    else: 
        return 0
"""    

def gyradius(x,y,z):
    gyr=np.sqrt(np.nanvar(x)+np.nanvar(y)+np.nanvar(z))
    return gyr


# concatenate row_wise matrices with different column length belonging to a dictionary
def concatenateMatrix(data,matrix):
    max_cols = max((entry[matrix].shape[1]) for entry in data.values())
    padded_matrices = [np.pad(entry[matrix], ((0, 0), (0, max_cols - entry[matrix].shape[1])), constant_values=np.nan) for entry in data.values()]
    result = np.vstack(padded_matrices)
    return result

#CORRECTION DES PARTICULES QUI SAUTENT D'UNE FRAME A UNE AUTRE 
# filled = correctGap(x, frame0, frame)
def correctGap(x, frame0, frame):
    # Initialize a NaN-filled array of the same length as full_range
    filled = np.full_like(frame, np.nan, dtype=np.float64)
    filled[(np.array(frame0) - frame0[0]).astype(int)]=x
    filled=np.array(filled) #convertit en tableau
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
    
    for traj in tracks.values():
       
        # Store drift values directly in the dictionary
        traj["xdrift"] = xdrift
        traj["ydrift"] = ydrift
        traj["zdrift"] = zdrift
        
        # Corrected coordinates based on drift
        time = traj["frame"]
        
        # it happens rarely but the time could be longer than the xdrift vector if particle appears during the film, this has to be trimmed
        
        time=time[time<len(xdrift)]
        
        traj["xcorr"] = traj["x"][:len(time)] - xdrift[time]+xdrift[0]
        traj["ycorr"] = traj["y"][:len(time)] - ydrift[time]+ydrift[0]
        traj["zcorr"] = traj["z"][:len(time)] - zdrift[time]+zdrift[0]
        
        traj["XCcorr"] = np.nanmean(traj["xcorr"])
        traj["YCcorr"] = np.nanmean(traj["xcorr"])
        traj["ZCcorr"] = np.nanmean(traj["zcorr"])
        
        
        # Calculate MSD (Mean Squared Displacement)
        #traj["MSD3corr"] = MSD3(traj["xcorr"], traj["ycorr"], traj["zcorr"])
        traj["MSDcorr"] = MSD2(traj["xcorr"], traj["ycorr"])
        
        # Calculate alpha coefficients
        traj["alpha2corr"] = alphaCoeff(traj["MSDcorr"])
        #traj["alpha3corr"] = alphaCoeff(traj["MSD3corr"])
        
        # Calculate diffusion coefficients
        slope2, _ = diffCoeff(data[selMovie]["dt"], traj["MSDcorr"])
        traj["D2corr"] = slope2 / 4
        # slope3, _ = diffCoeff(data[selMovie]["dt"], traj["MSD3corr"])
        # traj["D3corr"] = slope3 / 4
        #print(key)

    if doPlot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='2d')
        # Plot the 3D line
        ax.plot(xdrift, ydrift, zdrift, label='2D line')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_ylabel('Z')
        plt.title('data '+str(selMovie))
    # Return the modified tracks dictionary
    
    return(tracks, xdrift, ydrift, zdrift)

#Interpolation des MSD lorsque pris à diff temps 
def MSDinterp(data):
    for movie  in data:
        msdMean=np.nanmean(movie["MSDmat"],axis=0)
        tracks = movie["tracks"]
        timeMax = len(msdMean)*movie["dt"]
        timeOri = np.arange(0, timeMax, movie["dt"])   # time original
        timeInt = np.arange(0, max(timeOri), 10) # time where I want to sample
        interpolator = interp1d(timeOri, msdMean, kind='quadratic')  # Linear interpolation function
        MSDMeanInt = interpolator(timeInt) # MSD interpolated
        movie["timeOri"]=timeOri
        movie["MSDMeanOri"]=msdMean
        movie["timeInt"]=timeInt
        movie["MSDMeanInt"]=MSDMeanInt
    return (data)

def selectData(data, sel):
    for i, movie in enumerate(data):
        tracks=movie['tracks']
        selection = [k for k in tracks.keys() if tracks[k]['zone']==sel]


#CALCULATION OF ALPHA AND EFFECTIVE COEFF D 
def MSDtime(msd, dt, step=5):
    if (len(msd)<(step+1)):
        return np.array([]),np.array([])
    t = np.arange(1, len(msd))*dt 
    t = np.array(list(range(1,len(msd))))
    t = np.where(t <= 0, 1e-10, t)  # Replace 0 or negative values with a small number
    msd[1:] = np.where(msd[1:] <= 0, 1e-10, msd[1:]) # correct MSD=<0 by small number
    Deff=np.zeros(len(msd)-step-1)
    alpha=np.zeros(len(msd)-step-1)
    for i in range (0, len(msd)-step-1):
        slope, intercept = np.polyfit(np.log(t[i:i+step]), np.log(msd[i:i+step]), 1)
        Deff[i] = np.exp(intercept)
        alpha[i] = slope
    return alpha, Deff


#Calculation of the aspect ratio 
"""tracks.values() : récupère toutes les valeurs du dictionnaire tracks
"""


def asp_ratio(x,y):        
    #enlever les NaN que Python ne supporte pas 
    mask = np.isfinite(x) & np.isfinite(y)
    x_withoutnan = x[mask]
    y_withoutnan = y[mask]
    cov_matrix = np.cov(x_withoutnan, y_withoutnan) #covariance matrix 
    
    if (len(x_withoutnan)>5):
        #diagonalisation de la matrice 
        lambda_ratio, vecteurs_pro = np.linalg.eig(cov_matrix)
        lambda_1 = max(lambda_ratio) #variance max dans une direction 
        lambda_2 = min(lambda_ratio) #variance dans la direction perpendiculaire 
        aspect_ratio = (lambda_2)/(lambda_1)
    else:
        aspect_ratio = np.nan
        lambda_1 = np.nan
        lambda_2 = np.nan
    return aspect_ratio, lambda_1, lambda_2


#Filtering tracks, keeping only the tracks lambda_1 > 0.22 um (size of pixel)
def filter_tracks_by_lambda(tracks, lambda_min=0.22):
    """
    Filtre les tracks d'un movie où lambda_1 > lambda_min.
    
    Paramètres :
        tracks : dictionnaire de tracks (issu de import_tracks)
        lambda_min : seuil sur lambda_1 (défaut = 0.22)
    
    Retourne :
        filtered_tracks : sous-dictionnaire des tracks filtrés
    """
    filtered_tracks = {
        key: trk for key, trk in tracks.items()
        if trk["lambda_1"] > lambda_min
    }
    return filtered_tracks

def complex_modulusFFT(msd, dt, a):
    """
    Compute G′ and G″ from MSD using GSER.

    Parameters:
    - msd: array of mean squared displacement vs. lag time (m^2)
    - dt: time step between trajectory points (s)
    - a: particle radius (m)
    - T: temperature (K)

    Returns:
    - freqs: frequency array
    - G_elas: storage modulus
    - G_visc: loss modulus
    
    Use: 
        
    omega, G_elas, G_visc = complex_modulusFFT(msd, dt, a):
    plt.plot(omega, G1)
    
    
    """
    kB = 1.380649e-23  # Boltzmann constant
    T = 310 # 37 celsius
    N = len(msd)


    # # One-sided FT of MSD: integrate from 0 to inf
    # # rfft gives the non-negative frequency components
    #MSD after FT : from msd(lag_time)^2 to msd(omega)^2 
    msd_ft = np.fft.rfft(msd) * dt        # one-sided FT (scaled by dt)
    freqs  = np.fft.rfftfreq(N, d=dt)

    # Avoid division by zero at DC (freq=0)
    omega = 2 * np.pi * freqs[1:] #avoids index 0, starts at 1  
    msd_ft = msd_ft[1:]

    # GSER in Fourier space: G*(ω) = kT / (πa · iω · F(MSD)(ω))
    
    Gstar = kB * T / (np.pi * a * 1j * omega * msd_ft)

    G_elas = Gstar.real   # storage modulus
    G_visc = Gstar.imag   # loss modulus
    return omega, G_elas, G_visc


def complex_modulusLT(msd, dt, a, degree=5):
    """
    Fit MSD with a polynomial, then analytically Laplace transform it.
    Compute G*(s) from the Laplace-transformed MSD.

    Parameters:
    - t: time array
    - msd: mean squared displacement array
    - a: radius of the particle    
    - degree: degree of the polynomial fit

    Returns:
    - s: Laplace variable
    - msd_laplace: Laplace transform of the fitted MSD
    - G_star_laplace: complex modulus in Laplace domain

    Parameters:
    - s: Laplace variable
    - msd_laplace: Laplace transform of MSD
    - a: particle radius (m)
    - T: temperature (K)
    """
    # Fit MSD with a polynomial
    N = len(msd)
    t = dt*np.arange(N)
    coeffs = np.polyfit(t, msd, degree)
    poly = np.poly1d(coeffs)

    # Define symbolic variables
    tau, s = sym.symbols('tau s', real=True, positive=True)

    # Construct the polynomial symbolically
    poly_sym = 0
    for i, c in enumerate(coeffs):
        poly_sym += c * tau**i

    # Analytically Laplace transform the polynomial
    msd_laplace = sym.laplace_transform(poly_sym, tau, s, noconds=True)

     
    kB = 1.380649e-23  # Boltzmann constant
    T = 310 # 37 Celsius    
    G_star_laplace = (kB * T) / (np.pi * a * s * msd_laplace)
    
    
    # Prepare arrays for G' and G''
    mid = np.int32(N/2)
    freqs = fftfreq(N, dt)
    omega = 2 * np.pi * freqs
    omega = omega[1:mid]
    G1 = np.zeros_like(omega)
    G2 = np.zeros_like(omega)

    for i, w in enumerate(omega):
        # Substitute s = i*w
        G_star_iw = G_star_laplace.subs(s, 1j*w)
        # Extract real and imaginary parts
        G1[i] = sym.re(G_star_iw)
        G2[i] = -sym.im(G_star_iw)  # Note the minus sign for G''

    return omega, G1, G2



#Filtre : seuillage 
""" Permet de filtrer les points dans les data qui sont en dessous d'un certain seuil du MSD 
"""
def thrsh_MSD(MSD2Dmat) :
    mask_MSD = MSD2Dmat > 1e-1


# this increases the window
def MSDtime2(msd, dt, step=5):
    if (len(msd)<(step+1)):
        return np.array([]),np.array([])
    t = np.arange(1, len(msd))*dt 
    t=np.array(list(range(1,len(msd))))
    t = np.where(t <= 0, 1e-10, t)  # Replace 0 or negative values with a small number
    msd[1:] = np.where(msd[1:] <= 0, 1e-10, msd[1:]) # correct MSD=<0 by small number
    Deff=np.zeros(len(msd)-step-1)
    alpha=np.zeros(len(msd)-step-1)
    for i in range (0, len(msd)-step-1):
        slope, intercept = np.polyfit(np.log(t[1:i+step]), np.log(msd[1:i+step]), 1)
        Deff[i] = np.exp(intercept)
        alpha[i] = slope
    return alpha, Deff

def mean_MSD_filtered(movie):
    """Computes the mean MSD of all the filtered tracks
    
    Paramètres :
        movie : dictionnaire contenant "MSD2Dmat_filtered" de shape (N_filtered, max_len)
    
    Retourne :
        MSD_mean : array de shape (max_len,) — moyenne sur les trajectoires, ignore les NaN

    """
    MSD_mean_alltr = np.nanmean(movie["MSD2Dmat_filtered"], axis=0)
    
    return MSD_mean_alltr


def plotAlphaDeff(alpha, msd, Deff,step=5):
    N_msd = len(msd) #Longueur totale de l'array MSD (es. 37)
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
    
    
    
    