#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 14:19:03 2026

@author: xuan_nguyen

"""

import numpy as np;
import seaborn as sns;
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
from scipy.stats import pearsonr



# %%----------------------PLOT X, Y : schéma simplifié des particules -----------------------


"""
j : indice du fichier traité dans data
i : numéro de la trajectoire

"""

def plot_XYprojection(data):

    for j in range(len(data)):

        plt.figure()

        name_file = data[j]["name"]

        for i in range(len(data[j]["tracks"])):

            x1 = data[j]["tracks"][i]["x"]
            y1 = data[j]["tracks"][i]["y"]
            d = data[j]["tracks"][i]["D2"]

            if (d < .1) or (np.isnan(d)):
                plt.plot(x1, y1, c='red')
            else:
                plt.plot(x1, y1, c='blue')

        plt.title(name_file)
        plt.xlabel("x")
        plt.ylabel("y")

        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal', adjustable='box')

        plt.show()


# %% ---------------- PLOT MSD EN FONCTION DU TEMPS ------------------------------------


def plot_msd(data):

    for i in range(len(data)):

        fig, ax = plt.subplots(1, 2, layout='constrained')

        m = data[i]["MSD2Dmat_filtered"]
        frame_time = data[i]["timeOri"]
        t = data[i]["dt"] * np.arange(0, len(frame_time))
        name = data[i]["name"]

        fig.suptitle(name)

        # MSD vs time
        ax[0].plot(t, m.T)
        ax[0].set_xlabel("Time (sec)")
        ax[0].set_ylabel("MSD (um^2)")
        ax[0].set_title("MSD over time")

        # log-log MSD
        ax[1].plot(t[1:], m.T[1:]) #0 element is deleted since it is a log plot
        ax[1].set_xscale("log")
        ax[1].set_yscale("log")
        ax[1].set_xlabel("log Time (sec)")
        ax[1].set_ylabel("log MSD")
        ax[1].set_title("Log-Log")

        plt.show()
        
        
        # %% -----------------PLOT MSD trié avec alpha > 0 ----------------------------
        
        def plot_msd_filtered_alpha(data, alpha_min = 0.75):

            for i in range(len(data)):

                fig, ax = plt.subplots(1, 2, layout='constrained')

                alphas = np.array(data[i]["alpha2"])
                msd_all = np.array(data[i]["MSD2Dmat"])
                
                #Masque alpha > alpha_min 
                mask = alphas > alpha_min 
                msd_filtered = msd_all[mask]
                
                t = data[i]["dt"] * np.arange(msd_filtered.shape[1])
                name = data[i]["name"]

                fig.suptitle(name)

                # MSD vs time
                ax[0].plot(t, msd_filtered.T)
                ax[0].set_xlabel("Time (sec)")
                ax[0].set_ylabel("MSD (um^2)")
                ax[0].set_title("MSD over time")

                # log-log MSD
                ax[1].plot(t[1:], msd_filtered.T[1:]) #0 element is deleted since it is a log plot
                ax[1].set_xscale("log")
                ax[1].set_yscale("log")
                ax[1].set_xlabel("log Time (sec)")
                ax[1].set_ylabel("log MSD")
                ax[1].set_title("Log-Log")

                plt.show()

        
        
        # %% -------------------PLOT ALPHA VIOLIN PLOT MULTIPLE : CELUI UTILISE --------------
        

def plot_alpha_BT_multiple(data):
    all_alpha = []
    all_labels = []
    all_median = []
    all_colors = []
    palette_B = sns.color_palette("Blues", n_colors=10)
    palette_T = sns.color_palette("Oranges", n_colors=10)
    
    # Tri par zone puis par numéro de LN
    def extract_ln_number(d):
        return int(d["LN"].replace("LN", ""))
    data_sorted = sorted(data, key=lambda x: (x["zone"], extract_ln_number(x)))
    
    idx_B, idx_T = 0, 0
    for d in data_sorted:
        alpha_values = np.array(d["alpha2_filtered"])
        name = d["name"].replace(".csv", "")
        name = "_".join(name.split("_")[1:])
        
        all_alpha.append(alpha_values)
        all_labels.append(name)
        
        if d["zone"] == "B":
            all_colors.append(palette_B[idx_B % len(palette_B)])
            idx_B += 1
        else:
            all_colors.append(palette_T[idx_T % len(palette_T)])
            idx_T += 1
    
    plt.figure(figsize=(14, 6), dpi=150)
    
    # Violin plot
    sns.violinplot(
        data=all_alpha,
        inner="box",
        palette=all_colors,
        linewidth=1
    )
    
    # Stripplot
    sns.stripplot(
        data=all_alpha,
        jitter=True,
        color="black",
        size=3,
        alpha=0.5
    )
    
    # Calcul global pour les marges
    global_max = max([np.max(a) for a in all_alpha])
    global_min = min([np.min(a) for a in all_alpha])
    margin = 0.25 * (global_max - global_min)
    plt.ylim(global_min, global_max + margin)
    
    # Médiane
    for j, alpha_values in enumerate(all_alpha):
        median_alpha = np.median(alpha_values)
        all_median.append(median_alpha)
        plt.text(
            j,
            global_max + 0.1 * (global_max - global_min),
            f"{median_alpha:.2f}",
            ha='center',
            va='bottom',
            fontsize=8,
            color='red',
            fontweight='bold'
        )
    
    # Séparation B/T
    n_B = sum(1 for d in data_sorted if d["zone"] == "B")
    plt.axvline(n_B - 0.5, color='black', linestyle='--')
    
    # Titres de zone
    plt.text(n_B / 2 - 0.5, global_max + 0.2 * (global_max - global_min), 
             "Zone B", ha='center', fontsize=12)
    plt.text(n_B + (len(all_labels) - n_B) / 2 - 0.5, global_max + 0.2 * (global_max - global_min), 
             "Zone T", ha='center', fontsize=12)
    
    # Labels
    plt.xticks(
        ticks=np.arange(len(all_labels)),
        labels=[label[:10] + "..." if len(label) > 10 else label for label in all_labels],
        rotation=25,
        fontsize=8
    )
    
    plt.ylabel("Alpha")
    plt.title("Distribution de alpha dans la zone B et T")
    plt.xlabel("")
    plt.tight_layout()
    plt.show()
    print(all_median)
    
    
    
    # %% --------------------------------- PLOT DIFFUSION MULTIPLE ---------------------------
    
def diffusion_plot_multiple(data) :
    
    all_diffcoeff = []
    all_labels = []
    all_median = []
    
    palette_colors = sns.color_palette("Set2", n_colors=len(data))
    
    for i in range(len(data)):
        diffcoeff = np.array(data[i]["D2"])
        diffcoeff = diffcoeff[diffcoeff > 0]
        all_diffcoeff.append(diffcoeff)

            
        #NOM DU FICHIER 
        name = data[i]["name"].replace(".csv", "")
        name = "_".join(name.split("_")[1:]) #split à chaque _ et enlève le 1er élément (date)
        all_labels.append(name)

        # Meilleure qualité
    plt.figure(figsize=(10, 5), dpi=150)

        #Violin plot 
        # Violin + boxplot intégré
    sns.violinplot(
           data=all_diffcoeff,
           inner="box",
           palette=palette_colors,
           linewidth=1
       )
        
    sns.stripplot(
            data = all_diffcoeff, 
            jitter = True
            )
    
    plt.ylim(auto=True)
        
    global_max = max([np.max(d) for d in all_diffcoeff])
    global_min = min([np.min(d) for d in all_diffcoeff])

        #Affichage de la médiane 

    for j, diffcoeff in enumerate(all_diffcoeff):
            median_diff = np.median(diffcoeff) #calcul pour chaque fichier de la médiane 
            all_median.append(median_diff)
            
            plt.text(
            j,
            global_max + 0.05*(global_max - global_min),
            f"{median_diff:.2f}",
            ha='center',
            va='bottom',
            fontsize=8,
            color='red',
            fontweight='bold'
        )
            
       # Labels tronqués sous chaque violon
    plt.xticks(
        ticks=np.arange(len(all_labels)),
        labels=[label[:12] + "..." if len(label) > 12 else label for label in all_labels],
        rotation=20,
        fontsize=8
    )
            
    plt.ylabel("Coefficient diff (um^2/s)")
    plt.title("Distribution de D")
    plt.xlabel("")
    plt.show()

    print(all_median)
        
    
#%% ----------------------------- PLOT DIFFUSION / ONE FILE -----------------------------------

def diffusion_plot_1file(data):

    palette_colors = sns.color_palette("Set2", n_colors=1)
    
    i = 2
    
    diffcoeff = np.array(data[i]["D2"])
    print(len(diffcoeff))
    diffcoeff = diffcoeff[diffcoeff > 0.2] 
    print(len(diffcoeff))

    name = data[i]["name"].replace(".csv", "")
    name = "_".join(name.split("_")[1:])

    # 🔧 IMPORTANT → liste pour 1 seul violon
    sns.violinplot(
        data=[diffcoeff],
        inner="box",
        palette=palette_colors,
        linewidth=1
    )
        
    sns.stripplot(
        data=[diffcoeff],
        jitter=True
    )
    
    plt.ylim(auto=True)

    # 🔧 max/min corrects
    global_max = np.max(diffcoeff)
    global_min = np.min(diffcoeff)

    median_diff = np.median(diffcoeff)

    # 🔧 position correcte du texte
    plt.text(
        0,  # x position (centre du violon)
        global_max + 0.05*(global_max - global_min),
        f"{median_diff:.2f}",
        ha='center',
        va='bottom',
        fontsize=9,
        color='red',
        fontweight='bold'
    )

    # Label propre
    plt.xticks(
        [0],
        [name[:12] + "..." if len(name) > 12 else name],
        rotation=20,
        fontsize=8
    )

    plt.ylabel("Coefficient diff (um^2/s)")
    plt.title("Distribution de D")
    plt.xlabel("")

    plt.tight_layout()
    plt.show()

    print(median_diff)
      
    
    #%% ------------------ DIFFUSION PLOT MULTIPLE MELANGE ZONE B/ZONE T ----------------------

""" USED ON MAIN LOOP
"""

def diffusion_plot_BT_LN(data):

    all_diffcoeff = []
    all_labels = []
    all_median = []
    all_colors = []

    palette_B = sns.color_palette("Purples", n_colors=10)
    palette_T = sns.color_palette("Greens", n_colors=10)

    # Sort data : by LN number then by zone
    def extract_ln_number(d):
        return int(d["LN"].replace("LN", ""))

    data_sorted = sorted(data, key=lambda x: (x["zone"], extract_ln_number(x)))

    idx_B, idx_T = 0, 0

    # Make lists : 
    for d in data_sorted:

        diffcoeff = np.array(d["D2_filtered"])

        name = d["name"].replace(".csv", "")
        name = "_".join(name.split("_")[1:])

        all_diffcoeff.append(diffcoeff)
        all_labels.append(name)

        if d["zone"] == "B":
            all_colors.append(palette_B[idx_B % len(palette_B)])
            idx_B += 1
        else:
            all_colors.append(palette_T[idx_T % len(palette_T)])
            idx_T += 1

    plt.figure(figsize=(14, 6), dpi=150)

    # Violin plots
    sns.violinplot(
        data=all_diffcoeff,
        inner="box",
        palette=all_colors,
        linewidth=1
    )

    #Stripplot
    sns.stripplot(
        data=all_diffcoeff,
        jitter=True,
        color="black",
        size=3,
        alpha=0.5
    )

    plt.ylim(auto=True)

    # Medians
    global_max = max([np.max(d) for d in all_diffcoeff])
    global_min = min([np.min(d) for d in all_diffcoeff])

    for j, diffcoeff in enumerate(all_diffcoeff):
        median_diff = np.median(diffcoeff)
        all_median.append(median_diff)

        plt.text(
            j,
            global_max + 0.05*(global_max - global_min),
            f"{median_diff:.2f}",
            ha='center',
            va='bottom',
            fontsize=8,
            color='red',
            fontweight='bold'
        )

    # B/T zone separation
    n_B = sum(1 for d in data_sorted if d["zone"] == "B")
    plt.axvline(n_B - 0.5, color='black', linestyle='--')

    # Short labels
    plt.xticks(
        ticks=np.arange(len(all_labels)),
        labels=[label[:10] + "..." if len(label) > 10 else label for label in all_labels],
        rotation=25,
        fontsize=8
    )

    # Titles
    plt.text(n_B/2 - 0.5, global_max*1.15, "Zone B", ha='center', fontsize=12)
    plt.text(n_B + (len(all_labels)-n_B)/2 - 0.5, global_max*1.15, "Zone T", ha='center', fontsize=12)

    plt.ylabel("Coefficient diff (um^2/s)")
    plt.title("Distribution de D dans la zone B et T")
    plt.xlabel("")

    plt.tight_layout()
    plt.show()

    print(all_median)
    
    #%% STRIPPLOT ZONE B VS ZONE T (ALL SPOTS)
    
def stripplot_diffusion_BT(data):

    all_D = []
    all_zone = []
    all_LN = []

    # Extract data 
    for movie in data:
        diffcoeff = np.array(movie["D2"])
        diffcoeff = diffcoeff[diffcoeff > 0]  # enlever les valeurs invalides

        zone = movie["zone"]
        LN = movie["LN"]  # Utiliser le ganglion comme "hue"

        for val in diffcoeff:
            all_D.append(val)
            all_zone.append(zone)
            all_LN.append(LN)

    plt.figure(figsize=(8, 6), dpi=150)

    #Palette unique par ganglion
    unique_LN = list(set(all_LN))
    palette = sns.color_palette("hsv", len(unique_LN))
    couleur_LN = {ln: c for ln, c in zip(unique_LN, palette)}

    #Stripplot
    sns.stripplot(
        x=all_zone,
        y=all_D,
        hue=all_LN,
        palette=couleur_LN,
        jitter=True,
        size=4,
        alpha=0.7,
        order = ["B", "T"]
    )

    plt.ylabel("Coefficient diff (µm²/s)")
    plt.xlabel("Zone")
    plt.title("D distribution")

    #More compact legend
    plt.legend(
        title="Ganglion",
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=7
    )

    plt.tight_layout()
    plt.show()
    
    

    
   #%% DIFFUSION PLOT MULTIPLE MELANGE ZONE B/ZONE T FILTERED 

def diffusion_plot_BT_LN_filtered(data):

    all_diffcoeff = []
    all_labels = []
    all_median = []
    all_colors = []

    palette_B = sns.color_palette("Purples", n_colors=10)
    palette_T = sns.color_palette("Greens", n_colors=10)

    # Sort data : by LN number then by zone
    def extract_ln_number(d):
        return int(d["LN"].replace("LN", ""))

    data_sorted = sorted(data, key=lambda x: (x["zone"], extract_ln_number(x)))

    idx_B, idx_T = 0, 0

    # Make lists : 
    for d in data_sorted:

        diffcoeff = np.array(d["D2_filtered"])


        name = d["name"].replace(".csv", "")
        name = "_".join(name.split("_")[1:])

        all_diffcoeff.append(diffcoeff)
        all_labels.append(name)

        if d["zone"] == "B":
            all_colors.append(palette_B[idx_B % len(palette_B)])
            idx_B += 1
        else:
            all_colors.append(palette_T[idx_T % len(palette_T)])
            idx_T += 1

    plt.figure(figsize=(14, 6), dpi=150)

    # Violin plots
    sns.violinplot(
        data=all_diffcoeff,
        inner="box",
        palette=all_colors,
        linewidth=1
    )

    #Stripplot
    sns.stripplot(
        data=all_diffcoeff,
        jitter=True,
        color="black",
        size=3,
        alpha=0.5
    )

    plt.ylim(auto=True)

    # Medians
    global_max = max([np.max(d) for d in all_diffcoeff])
    global_min = min([np.min(d) for d in all_diffcoeff])

    for j, diffcoeff in enumerate(all_diffcoeff):
        median_diff = np.median(diffcoeff)
        all_median.append(median_diff)

        plt.text(
            j,
            global_max + 0.05*(global_max - global_min),
            f"{median_diff:.2f}",
            ha='center',
            va='bottom',
            fontsize=8,
            color='red',
            fontweight='bold'
        )

    # B/T zone separation
    n_B = sum(1 for d in data_sorted if d["zone"] == "B")
    plt.axvline(n_B - 0.5, color='black', linestyle='--')

    # Short labels
    plt.xticks(
        ticks=np.arange(len(all_labels)),
        labels=[label[:10] + "..." if len(label) > 10 else label for label in all_labels],
        rotation=25,
        fontsize=8
    )

    # Titles
    plt.text(n_B/2 - 0.5, global_max*1.15, "Zone B", ha='center', fontsize=12)
    plt.text(n_B + (len(all_labels)-n_B)/2 - 0.5, global_max*1.15, "Zone T", ha='center', fontsize=12)

    plt.ylabel("Coefficient diff (um^2/s)")
    plt.title("Distribution de D dans la zone B et T")
    plt.xlabel("")

    plt.tight_layout()
    plt.show()

    print(all_median)
    
    #%% STRIPPLOT ZONE B VS ZONE T : MOYENNES 
    
def stripplot_moyenne_diffusion(data):
    """
    Pour chaque fichier (movie), calcule la moyenne de diffusion D2,
    et trace un stripplot avec x = zone (B/T) et y = diffusion moyenne.
    La couleur correspond au ganglion (LN).
    """
    moy_D = []
    zone_list = []
    LN_list = []

    # Calculer la moyenne par fichier
    for movie in data:
        diffcoeff = np.array(movie["D2"])
        diffcoeff = diffcoeff[diffcoeff > 0]  # enlever valeurs invalides

        if len(diffcoeff) == 0:
            continue  # éviter les fichiers sans données valides

        moyenne = np.mean(diffcoeff)

        moy_D.append(moyenne)
        zone_list.append(movie["zone"])
        LN_list.append(movie["LN"])

    #Créer une palette par ganglion
    unique_LN = list(set(LN_list))
    palette = sns.color_palette("hsv", len(unique_LN))
    couleur_LN = {ln: c for ln, c in zip(unique_LN, palette)}

    plt.figure(figsize=(8, 6), dpi=150)

    #Stripplot
    sns.stripplot(
        x=zone_list,
        y=moy_D,
        hue=LN_list,
        palette=couleur_LN,
        jitter=True,
        size=6,
        alpha=0.8,
        order=["B", "T"]  # B avant T
    )

    plt.ylabel("Diffusion moyenne D (µm²/s)")
    plt.xlabel("Zone")
    plt.title("Diffusion moyenne par fichier et zone")

    plt.legend(
        title="Ganglion",
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=7
    )
    
    
   #%% ALPHA PLOT MULTIPLE : ZONE B/ZONE T 
   
   
    
    #%% DIFFUSION EN FONCTION DE ALPHA 
    
def diff_alpha(data) : 
    i = 0 #number of the file 
    alpha_values = data[i]["alpha2"]
    diff_values = data[i]["D2"]
    print(f"Valeur maximum de diffusion est {max(diff_values)}, indice est {np.argmax(diff_values)}")
    
    sns.stripplot(x = alpha_values, y = diff_values)
    plt.ylabel("Coefficient diffusion (um^2/s")
    plt.xlabel("Alpha")
    
    
    
    #%% TEST DE CORRELATION PEARSON : ALPHA/DIFFUSION

def correlation_alpha_diff(data):

    all_alpha = []
    all_diff = []

    # Récupérer toutes les valeurs
    for i in range(len(data)):
        alpha = np.array(data[i]["alpha2"])
        diff = np.array(data[i]["D2"])

        # enlever NaN
        mask = np.isfinite(alpha) & np.isfinite(diff)
        alpha = alpha[mask]
        diff = diff[mask]

        all_alpha.extend(alpha)
        all_diff.extend(diff)

    all_alpha = np.array(all_alpha)
    all_diff = np.array(all_diff)

    # Corrélation de Pearson
    r, p_value = pearsonr(all_diff, all_alpha)

    # Plot
    plt.figure(figsize=(6, 5), dpi=150)
    plt.scatter(all_diff, all_alpha, alpha=0.6)

    plt.xlabel("Alpha")
    plt.ylabel("D (um²/s)")
    plt.title(f"Corrélation alpha vs D\nr = {r:.2f}, p = {p_value:.2e}")

    plt.tight_layout()
    plt.show()

    print(f"Pearson r = {r}")
    print(f"p-value = {p_value}")
    
    
    #%% MSD filtered 
    
    def plot_msd_filtered(data):

            i = 0

            fig, ax = plt.subplots(1, 2, layout='constrained')

            m = data[i]["MSD2Dmat_filtered"]
            t = data[i]["dt"] * np.arange(1, m.shape[1] + 1) 
            name = data[i]["name"]
            
            mask = t <= 5
            t_cut = t[mask]
            m_cut = m[:, mask] 

            fig.suptitle(name)

            # MSD vs time
            ax[0].plot(t_cut, m_cut.T)
            ax[0].set_xlabel("Time (sec)")
            ax[0].set_ylabel("MSD (um^2)")
            ax[0].set_title("MSD over time")

            # log-log MSD
            ax[1].plot(t[1:], m.T[1:]) #0 element is deleted since it is a log plot
            ax[1].set_xscale("log")
            ax[1].set_yscale("log")
            ax[1].set_xlabel("log Time (sec)")
            ax[1].set_ylabel("log MSD")
            ax[1].set_title("Log-Log")

            plt.show()
            
            
    #%% MSD MEAN FILTERED 
       
def mean_MSD_time(data) : 
       
    for i in range(len(data)): 
        m = data[i]["MSD2Dmeanfiltered"]
        t = data[i]    
        
        
    
    #%% diffusion D en fonction de lambda 
    
    i = 0 #file index 
    diffusion_D = np.array(data[i]["D2"])
    lambda_values = np.array(data[i]["lambda_1"])
    mask = lambda_values > 0.22 #displacement of particled > size of the pixel 
    lambda_filtered = lambda_values[mask]
    diffusion_D_filtered = diffusion_D[:, mask] if diffusion_D.ndim == 2 else diffusion_D[mask]
    
    plt.scatter(lambda_filtered, diffusion_D_filtered.T)
    
    
    #%% diffusion D en fonction de lambda 
    
    i = 0 #file index 
    diffusion_D = np.array(data[i]["D2_filtered"])
    lambda_values = np.array(data[i]["lambda_1_filtered"])
    
    plt.scatter(lambda_filtered, diffusion_D_filtered.T)
    
    #%% G and G'' calculated for every file 
    
def g1_g2(data) :
    
    fig, ax = plt.subplots(1, 2, layout='constrained')
    
    for i in range(len(data)) : 
        G1 = data[i]["G_elastic"]
        G2 = data[i]["G_viscous"]
        omega = data[i]["omega"]
        name = data[i]["name"]
        
        fig.subtitle(name)

   #G_elastic as a function of omega 
   
        ax[0].plot(omega, G1)
        ax[0].set_xlabel("Omega (Hz ?)")
        ax[0].set.ylabel("G1 (Pa)")
        ax[0].set_title("Elastic modulus")
        
    # G loss as a function of omega 
    
        ax[1].plot(omega, G2)
        ax[1].set_xlabel("Omega (Hz?)")
        ax[1].set.ylabel("G2 (Pa)")
        ax[1].set_title("Loss modulus")
        
        plt.show()
    
    #%% SCATTERPLOT G' and G''
    
"""
i = 0
viscoel_modulus = data[i]["G_elastic"]

# Création d'un DataFrame pour seaborn
df = pd.DataFrame({"G_elastic": viscoel_modulus})

fig, ax = plt.subplots()

# Violin plot + strip plot (points individuels) + médiane
sns.violinplot(data=df, y="G_elastic", ax=ax, inner="quart")  # inner="quart" affiche médiane + quartiles
sns.stripplot(data=df, y="G_elastic", ax=ax, color="black", alpha=0.3, size=3)  # points individuels

ax.set_yscale("log")
ax.set_ylabel("G' (Pa)")
ax.set_title(data[i]["name"])
plt.show()
"""
    
 #%% PLOT NOMBRE DE POINTS ET RAYON DE GYRATION PAR TRAJECTOIRE 

#file index
"""
gyr_values = np.array([trk["gyradius"] for trk in tracks.values()])

x_spots = np.arange(len(gyr_values)) + 1  # numéro des spots

plt.figure()
plt.scatter(gyr_values, x_spots)

plt.xlabel("Nombre de spots")
plt.ylabel("Rayon de gyration")
plt.title("Rayon de gyration par spot")

plt.show()
    
"""
# %%----------------------PICKLE-----------------------

# save dictionary
""""import pickle
from pathlib import Path    
path = Path(directory_path)
cond=path.parent.name
pkl_path = path / f"{cond}.pkl"  # on définit pkl_path pour la lecture
with open(directory_path+'/'+cond+'.pkl', 'wb') as file:
    # Serialize and save the dictionary to the file
    pickle.dump(data, file) #transforme data en format binaire
    
 #Réouverture
root = tk.Tk()
root.withdraw()  # Hide the root window
pkl_path = filedialog.askopenfilename() #dialogue 

with open(pkl_path, "rb") as file :
    data_loaded = pickle.load(file)
    
    
print("Type:", type(data_loaded))
print('flag3')"""

