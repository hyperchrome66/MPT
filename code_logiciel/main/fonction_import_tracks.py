#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 16:29:53 2026

@author: xuan_nguyen
"""


# %% -- IMPORT TRACKS -- 


import pandas as pd;

import numpy as np;
import matplotlib.pyplot as plt;
from fonctions_MPT import (
   MSD2, 
   correctGap,
   diffCoeff, 
   gyradius, 
   alphaCoeff, 
   asp_ratio,
   complex_modulusFFT,
   confinement_radius
   )

        
"""Importation du fichier
df : dataframe
df.POSITION_X (par exemple) : prend la colonne POSITION_X du dataframe
to_numpy() : convertit la colonne en array NUMPY
x, y, z, t, r, q, f: contenu sous forme de tableaux NumPY (array)

CREATION DU DICO : tracks = {}
tr : contient le nombre de trajectoires récupérées (à partir du nombre de part. différentes)
np.where((df.TRACK_ID)==i) : dans l'array de la colonne en question, sélectionne les trajectoires
si l'ID est égal à i, récupère les positions x, y, z etc de la parti. ID=i

"""


def import_tracks(name):
    df=pd.read_csv(name,skiprows=[1,2,3], encoding='ISO-8859-1');
    print(df.POSITION_T.dtype)
    tr=np.unique(df.TRACK_ID); #récupère tous les numéros des tracks différents, les trie 
    tr=np.int16(tr[~np.isnan(tr)])
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
        
        #OBTENTION DE X, Y, Z, T, R, Q, F EN TABLEAUX SELON ID 
        x=allx[np.where((df.TRACK_ID)==i)]; #cherche tous les x correspond à ID=i 
        y=ally[np.where((df.TRACK_ID)==i)]; #cherche tous les y correspond à ID=i 
        z=allz[np.where((df.TRACK_ID)==i)]; #cherche tous les Z correspond à ID=i 
        t=allt[np.where((df.TRACK_ID)==i)]; #cherche tous les t correspond à ID=i 
        r=allr[np.where((df.TRACK_ID)==i)];
        q=allq[np.where((df.TRACK_ID)==i)];
        frame0=allf[np.where((df.TRACK_ID)==i)] # count from 0!!!
        
        #CALCUL DU TEMPS
        #np.argot(t) : ordonne le tableau t en fonction du temps t 
        a=np.argsort(t) #a renvoie la liste des indices dans l'ordre en fct de t
        t=t[a] #tableau t reclassé avec les indices reclassés avec a 
        dt=np.float64(np.diff(t[0:2])[0]) #dt = t(frame1) - t(frame0)
        frame=np.arange(min(frame0),max(frame0)+1,1) #réordonne les frames
        t = frame*dt
        
        #CORRECTION DES SAUTS DE PARTICULES   
        x= correctGap(x[a], frame0, frame) 
        y= correctGap(y[a], frame0, frame)
        z= correctGap(z[a], frame0, frame)
        r= correctGap(r[a], frame0, frame)
        q= correctGap(q[a], frame0, frame)
        
        #np.nanmean : calcule la moyenne arithm, ignore NaN 
        #retourne la moyenne des éléments du tableau 
        xmean=np.nanmean(x); 
        ymean=np.nanmean(y);
        zmean=np.nanmean(z);
        rmean=np.nanmean(r)
        qmean=np.nanmean(q)
        ngap= sum(np.isnan(x)) #somme du nombre de sauts de frames 
        
        #CALCUL DU MSD 
        #Appel des fonctions MSD2, alphaCoeff, diffCoeff, gyradius, asp_ratio, confinement_radius
        msd2D = MSD2(x,y); 
        alpha2 = alphaCoeff(msd2D);        
        slope, err2 = diffCoeff(dt, msd2D); 
        D2 = slope/4; #diffusion à 2 dimensions 
        gyr = gyradius(x,y,z);
        aspect_ratio, lambda_1, lambda_2 = asp_ratio(x, y)
        pore_size = confinement_radius(msd2D, dt,  a=0.255e-6)

        #alphaTime, DeffTime = MSDtime(msd3D, dt, 10)  # sliding window fit alpha dn Deff (3D only)

        #CREATION D'UN DICO POUR LA PARTICULE SI NOMBRE DE POSITIONS > 5 
        if (len(x)>=5):
            tracks[k]={
                "id": i, 
                "x": x, "y": y,  "z": z, "t": t, "frame": frame, "radius":r, "quality":q,
                "dt":dt, "nspots": len(x),"xmean": xmean, "ymean": ymean,  "zmean": zmean, "rmean": rmean,
                "qmean": qmean, "ngap": ngap,
                "MSD2D": msd2D, "alpha2": alpha2, "D2": D2, "err2": err2, "gyradius": gyr, 
                "aspect_ratio" : aspect_ratio, "lambda_1" : lambda_1, "lambda_2" : lambda_2,
                "pore_size" : pore_size
                }
            k=k+1;
        print(k) #affiche le nombre de trajectoires
    return tracks

