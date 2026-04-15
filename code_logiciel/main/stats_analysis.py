#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 14:32:50 2026

@author: xuan_nguyen
"""

import numpy as np
from scipy.stats import mannwhitneyu, shapiro, ttest_ind
import seaborn as sns;
import matplotlib.pyplot as plt

#%% Comparaison zone contrôle (dans l'eau) vs zone B/zone T 

def ctrlvsLN(data, variable = "D2_filtered"):
    
    """Compare les distributions de la zone B ou zone T vs zone contrôle dans l'eau avec un test
    de Mann Whitney. 
      
    Paramètres :
        data     : liste de movies
        variable : clé du dictionnaire à comparer (ex: "alpha2_filtered", "D2_filtered")
        """
        
    #Separate data by zone   
    
    values_B = np.concatenate([d[variable] for d in data if d["zone"] == "B"])
    values_B = values_B[~np.isnan(values_B)]
    values_T = np.concatenate([d[variable] for d in data if d["zone"] == "T"])
    values_T = values_T[~np.isnan(values_T)]
    values_CTRL = np.concatenate([d[variable] for d in data if d["zone"] == "No LN"])
    values_CTRL = values_CTRL[~np.isnan(values_CTRL)]
    
    print(f"=== Comparaison {variable} : contrôle vs zone B ===")
    print(f"N Contrôle : {len(values_CTRL)} particules")
    print(f"N Zone B : {len(values_B)} particules")
    print(f"Médiane Contrôle : {np.median(values_CTRL):.4f}")
    print(f"Médiane Zone B : {np.median(values_B):.4f}")
    print(f"Moyenne Contrôle : {np.mean(values_CTRL):.4f}")
    print(f"Moyenne Zone B : {np.mean(values_B):.4f}")
    
    # Test de normalité (Shapiro-Wilk)
    _, p_shapiro_CTRL = shapiro(values_CTRL)
    _, p_shapiro_B = shapiro(values_B)
    print("\nTest de normalité Shapiro-Wilk :")
    print(f"  CTRL : p = {p_shapiro_CTRL:.2e} {'→ normale' if p_shapiro_CTRL > 0.05 else '→ NON normale'}")
    print(f"  Zone B : p = {p_shapiro_B:.2e} {'→ normale' if p_shapiro_B > 0.05 else '→ NON normale'}")
    
    # Test de Mann-Whitney U
    stat, p_value = mannwhitneyu(values_CTRL, values_B, alternative='two-sided')
    print("\nTest de Mann-Whitney U :")
    print(f"  Statistique U : {stat:.4f}")
    print(f"  p-value       : {p_value:.2e}")
    if p_value < 0.05:
        print("  → Différence SIGNIFICATIVE entre CTRL et zone B (p < 0.05)")
    else:
        print("  → Pas de différence significative entre CTRL et zone B (p > 0.05)")
        
    #Contrôle vs Zone T 
    print("")
    print(f"=== Comparaison {variable} : contrôle vs zone T ===")
    print(f"N Contrôle : {len(values_CTRL)} particules")
    print(f"N Zone T : {len(values_T)} particules")
    print(f"Médiane Contrôle : {np.median(values_CTRL):.4f}")
    print(f"Médiane Zone T : {np.median(values_T):.4f}")
    print(f"Moyenne Contrôle : {np.mean(values_CTRL):.4f}")
    print(f"Moyenne Zone T : {np.mean(values_T):.4f}")
    
    # Test de normalité (Shapiro-Wilk)
    _, p_shapiro_CTRL = shapiro(values_CTRL)
    _, p_shapiro_T = shapiro(values_T)
    print("\nTest de normalité Shapiro-Wilk :")
    print(f"  CTRL : p = {p_shapiro_CTRL:.2e} {'→ normale' if p_shapiro_CTRL > 0.05 else '→ NON normale'}")
    print(f"  Zone B : p = {p_shapiro_T:.2e} {'→ normale' if p_shapiro_T > 0.05 else '→ NON normale'}")
    
    # Test de Mann-Whitney U
    stat, p_value = mannwhitneyu(values_CTRL, values_T, alternative='two-sided')
    print("\nTest de Mann-Whitney U :")
    print(f"  Statistique U : {stat:.4f}")
    print(f"  p-value       : {p_value:.2e}")
    if p_value < 0.05:
        print("  → Différence SIGNIFICATIVE entre CTRL et zone T (p < 0.05)")
    else:
        print("  → Pas de différence significative entre CTRL et zone T (p > 0.05)")
    
    return values_CTRL, values_B, values_T, p_value


#%%Comparison test between B and T (arrays without nan)

def stats_BT(data, variable="D2_filtered"):
    """
    Compare les distributions de la zone B et T avec un test de Mann-Whitney ou Student
    selon la normalité des distributions.
    
    Paramètres :
        data     : liste de movies
        variable : clé du dictionnaire à comparer (ex: "alpha2_filtered", "D2_filtered")
    """
    # Séparer les données par zone
    values_B = np.concatenate([d[variable] for d in data if d["zone"] == "B"])
    values_T = np.concatenate([d[variable] for d in data if d["zone"] == "T"])
    
    print(f"=== Comparaison {variable} : Zone B vs Zone T ===")
    print(f"N Zone B : {len(values_B)} particules")
    print(f"N Zone T : {len(values_T)} particules")
    print(f"Médiane Zone B : {np.median(values_B):.4f}")
    print(f"Médiane Zone T : {np.median(values_T):.4f}")
    print(f"Moyenne Zone B : {np.mean(values_B):.4f}")
    print(f"Moyenne Zone T : {np.mean(values_T):.4f}")
    
    # Test de normalité (Shapiro-Wilk)
    _, p_shapiro_B = shapiro(values_B)
    _, p_shapiro_T = shapiro(values_T)
    both_normal = p_shapiro_B > 0.05 and p_shapiro_T > 0.05
    print("\nTest de normalité Shapiro-Wilk :")
    print(f"  Zone B : p = {p_shapiro_B:.2e} {'→ normale' if p_shapiro_B > 0.05 else '→ NON normale'}")
    print(f"  Zone T : p = {p_shapiro_T:.2e} {'→ normale' if p_shapiro_T > 0.05 else '→ NON normale'}")
    
    # Choix du test selon la normalité
    if both_normal:
        print("\n→ Les deux distributions sont normales : Test de Student")
        stat, p_value = ttest_ind(values_B, values_T, alternative='two-sided')
        print(f"  Statistique t : {stat:.4f}")
    else:
        print("\n→ Au moins une distribution NON normale : Test de Mann-Whitney U")
        stat, p_value = mannwhitneyu(values_B, values_T, alternative='two-sided')
        print(f"  Statistique U : {stat:.4f}")
    
    print(f"  p-value       : {p_value:.2e}")
    if p_value < 0.05:
        print("  → Différence SIGNIFICATIVE entre B et T (p < 0.05)")
    else:
        print("  → Pas de différence significative entre B et T (p > 0.05)")
    
    return values_B, values_T, p_value

#%% Comparison tests with arrays containing nan

def stats_nan_BT(data, variable="pore_size_filtered"):
    """
    Compare les distributions de la zone B et T avec un test de Mann-Whitney ou Student
    selon la normalité des distributions.
    
    Paramètres :
        data     : liste de movies
        variable : clé du dictionnaire à comparer (ex: "alpha2_filtered", "D2_filtered")
    """
    # Séparer les données par zone
    values_B = np.concatenate([d[variable] for d in data if d["zone"] == "B"])
    values_B = values_B[~np.isnan(values_B)]
    values_T = np.concatenate([d[variable] for d in data if d["zone"] == "T"])
    values_T = values_T[~np.isnan(values_T)]
    
    print(f"=== Comparaison {variable} : Zone B vs Zone T ===")
    print(f"N Zone B : {len(values_B)} particules")
    print(f"N Zone T : {len(values_T)} particules")
    print(f"Médiane Zone B : {np.median(values_B):.4f}")
    print(f"Médiane Zone T : {np.median(values_T):.4f}")
    print(f"Moyenne Zone B : {np.mean(values_B):.4f}")
    print(f"Moyenne Zone T : {np.mean(values_T):.4f}")
    
    # Test de normalité (Shapiro-Wilk)
    _, p_shapiro_B = shapiro(values_B)
    _, p_shapiro_T = shapiro(values_T)
    both_normal = p_shapiro_B > 0.05 and p_shapiro_T > 0.05
    print("\nTest de normalité Shapiro-Wilk :")
    print(f"  Zone B : p = {p_shapiro_B:.2e} {'→ normale' if p_shapiro_B > 0.05 else '→ NON normale'}")
    print(f"  Zone T : p = {p_shapiro_T:.2e} {'→ normale' if p_shapiro_T > 0.05 else '→ NON normale'}")
    
    # Choix du test selon la normalité
    if both_normal:
        print("\n→ Les deux distributions sont normales : Test de Student")
        stat, p_value = ttest_ind(values_B, values_T, alternative='two-sided')
        print(f"  Statistique t : {stat:.4f}")
    else:
        print("\n→ Au moins une distribution NON normale : Test de Mann-Whitney U")
        stat, p_value = mannwhitneyu(values_B, values_T, alternative='two-sided')
        print(f"  Statistique U : {stat:.4f}")
    
    print(f"  p-value       : {p_value:.2e}")
    if p_value < 0.05:
        print("  → Différence SIGNIFICATIVE entre B et T (p < 0.05)")
    else:
        print("  → Pas de différence significative entre B et T (p > 0.05)")
    
    return values_B, values_T, p_value

#%% comparison ctrl vs stimulated with arrays without nan 

def stats_condition(data, variable="D2_filtered"):
    """
    Compare les distributions d'une variable entre condition Control et Stimulated,
    séparément pour la zone B et la zone T.
    
    Paramètres :
        data     : liste de movies
# =============================================================================
#         variable : clé du dictionnaire à comparer
# =============================================================================
    """
    for zone in ["B", "T"]:
        print(f"\n{'='*50}")
        print(f"Zone {zone} — {variable}")
        print(f"{'='*50}")
        
        # Séparer par condition
        values_ctrl = np.concatenate([d[variable] for d in data 
                                      if d["zone"] == zone and d["condition"] == "Control"])
        values_ctrl = values_ctrl[~np.isnan(values_ctrl)]
        values_stim = np.concatenate([d[variable] for d in data 
                                      if d["zone"] == zone and d["condition"] == "Stimulated"])
        values_stim = values_stim[~np.isnan(values_stim)]
        
        print(f"N Control    : {len(values_ctrl)} particules")
        print(f"N Stimulated : {len(values_stim)} particules")
        print(f"Médiane Control    : {np.median(values_ctrl):.4f}")
        print(f"Médiane Stimulated : {np.median(values_stim):.4f}")
        print(f"Moyenne Control    : {np.mean(values_ctrl):.4f}")
        print(f"Moyenne Stimulated : {np.mean(values_stim):.4f}")
        
        # Test de normalité
        _, p_shapiro_ctrl = shapiro(values_ctrl)
        _, p_shapiro_stim = shapiro(values_stim)
        
        print(f"\nTest de normalité Shapiro-Wilk :")
        print(f"  Control    : p = {p_shapiro_ctrl:.2e} {'→ normale' if p_shapiro_ctrl > 0.05 else '→ NON normale'}")
        print(f"  Stimulated : p = {p_shapiro_stim:.2e} {'→ normale' if p_shapiro_stim > 0.05 else '→ NON normale'}")
        
        # Test de Mann-Whitney U
        stat, p_value = ttest_ind(values_ctrl, values_stim, alternative='two-sided')
        print(f"\nTest de Student :")
        print(f"  Statistique U : {stat:.4f}")
        print(f"  p-value       : {p_value:.2e}")
        if p_value < 0.05:
            print(f"  → Différence SIGNIFICATIVE entre Control et Stimulated (p < 0.05)")
        else:
            print(f"  → Pas de différence significative entre Control et Stimulated (p > 0.05)")
            
            #%% comparison ctrl vs stimulated with arrays containing nan

def stats_condition_nan(data, variable="pore_size_filtered"):
    """
    Compare les distributions d'une variable entre condition Control et Stimulated.
    
    Paramètres :
        data     : liste de movies
        variable : clé du dictionnaire à comparer
    """
    # Séparer par condition et filtrer les NaN
    values_ctrl = np.concatenate([d[variable] for d in data if d["condition"] == "Control"])
    values_stim = np.concatenate([d[variable] for d in data if d["condition"] == "Stimulated"])
    
    values_ctrl = values_ctrl[~np.isnan(values_ctrl)]
    values_stim = values_stim[~np.isnan(values_stim)]
    
    print(f"N Control    : {len(values_ctrl)} particules")
    print(f"N Stimulated : {len(values_stim)} particules")
    print(f"Médiane Control    : {np.nanmedian(values_ctrl):.4f}")
    print(f"Médiane Stimulated : {np.nanmedian(values_stim):.4f}")
    print(f"Moyenne Control    : {np.nanmean(values_ctrl):.4f}")
    print(f"Moyenne Stimulated : {np.nanmean(values_stim):.4f}")
    
    # Test de normalité
    _, p_shapiro_ctrl = shapiro(values_ctrl)
    _, p_shapiro_stim = shapiro(values_stim)
    both_normal = p_shapiro_ctrl > 0.05 and p_shapiro_stim > 0.05
    print(f"\nTest de normalité Shapiro-Wilk :")
    print(f"  Control    : p = {p_shapiro_ctrl:.2e} {'→ normale' if p_shapiro_ctrl > 0.05 else '→ NON normale'}")
    print(f"  Stimulated : p = {p_shapiro_stim:.2e} {'→ normale' if p_shapiro_stim > 0.05 else '→ NON normale'}")
    
    # Choix du test selon normalité
    if both_normal:
        print("\n→ Les deux distributions sont normales : Test de Student")
        stat, p_value = ttest_ind(values_ctrl, values_stim, alternative='two-sided')
        print(f"  Statistique t : {stat:.4f}")
    else:
        print("\n→ Au moins une distribution NON normale : Test de Mann-Whitney U")
        stat, p_value = mannwhitneyu(values_ctrl, values_stim, alternative='two-sided')
        print(f"  Statistique U : {stat:.4f}")
    
    print(f"  p-value : {p_value:.2e}")
    if p_value < 0.05:
        print("  → Différence SIGNIFICATIVE entre Control et Stimulated (p < 0.05)")
    else:
        print("  → Pas de différence significative entre Control et Stimulated (p > 0.05)")
    
    return values_ctrl, values_stim, p_value

#%%Graph de D2 séparé par zone B/T et condition contrôle/stimulée 

def plot_D2_condition_BT(data, variable="D2_filtered", ylabel="D2 (µm²/s)"):
    """
    Violin plot de D2 (ou autre variable) séparé par zone B/T et condition Control/Stimulated.
    """
    all_values = []
    all_labels = []
    all_colors = []
    all_median = []
    
    palette_B_ctrl = sns.color_palette("Purples", n_colors=5)[0]   # violet moyen
    palette_B_stim = sns.color_palette("Purples", n_colors=5)[2]   # violet foncé
    palette_T_ctrl = sns.color_palette("Greens",  n_colors=5)[0]   # vert moyen
    palette_T_stim = sns.color_palette("Greens",  n_colors=5)[2]   # vert foncé
    
    # Ordre : B_Control, B_Stimulated, T_Control, T_Stimulated
    groups = [
        ("B", "Control",    palette_B_ctrl, "B Control"),
        ("B", "Stimulated", palette_B_stim, "B Stimulated"),
        ("T", "Control",    palette_T_ctrl, "T Control"),
        ("T", "Stimulated", palette_T_stim, "T Stimulated"),
    ]
    
    for zone, condition, color, label in groups:
        values = np.concatenate([d[variable] for d in data
                                 if d["zone"] == zone and d["condition"] == condition])
        if len(values) > 0:
            all_values.append(values)
            all_labels.append(label)
            all_colors.append(color)
    
    plt.figure(figsize=(10, 6), dpi=150)
    
    # Violin plot
    sns.violinplot(
        data=all_values,
        inner="box",
        palette=all_colors,
        linewidth=1
    )
    
    # Stripplot
    sns.stripplot(
        data=all_values,
        jitter=True,
        color="black",
        size=3,
        alpha=0.5
    )
    
    # Calcul global pour les marges
    global_max = max([np.max(v) for v in all_values])
    global_min = min([np.min(v) for v in all_values])
    margin = 0.25 * (global_max - global_min)
    plt.ylim(global_min, global_max + margin)
    
    # Médiane
    for j, values in enumerate(all_values):
        median_val = np.median(values)
        all_median.append(median_val)
        plt.text(
            j,
            global_max + 0.1 * (global_max - global_min),
            f"{median_val:.3f}",
            ha='center',
            va='bottom',
            fontsize=8,
            color='red',
            fontweight='bold'
        )
    
    # Séparation B/T
    n_B = sum(1 for zone, condition, _, _ in groups
              if zone == "B" and any(
                  d["zone"] == zone and d["condition"] == condition
                  for d in data))
    plt.axvline(n_B - 0.5, color='black', linestyle='--')
    
    # Titres de zone
    plt.text(n_B / 2 - 0.5, global_max + 0.2 * (global_max - global_min),
             "Zone B", ha='center', fontsize=12, fontweight='bold')
    plt.text(n_B + (len(all_values) - n_B) / 2 - 0.5, global_max + 0.2 * (global_max - global_min),
             "Zone T", ha='center', fontsize=12, fontweight='bold')
    
    # Labels
    plt.xticks(
        ticks=np.arange(len(all_labels)),
        labels=all_labels,
        rotation=15,
        fontsize=9
    )
    
    plt.ylabel(ylabel)
    plt.title(f"Distribution de {variable} par zone et condition")
    plt.xlabel("")
    plt.tight_layout()
    plt.show()
    print("Médianes :", dict(zip(all_labels, all_median)))
    
#%%Graph de pore size séparé par zone B/T et condition contrôle/stimulée 


def plot_poresize_condition_BT(data, variable="pore_size_filtered", ylabel="Pore size (µm)"):
    """
    Violin plot du pore size (ou autre variable) séparé par zone B/T et condition Control/Stimulated.
    """
    all_values = []
    all_labels = []
    all_colors = []
    all_median = []
    
    palette_B_ctrl = sns.color_palette("Purples", n_colors=5)[0]   # violet moyen
    palette_B_stim = sns.color_palette("Purples", n_colors=5)[2]   # violet foncé
    palette_T_ctrl = sns.color_palette("Greens",  n_colors=5)[0]   # vert moyen
    palette_T_stim = sns.color_palette("Greens",  n_colors=5)[2]   # vert foncé
    
    # Ordre : B_Control, B_Stimulated, T_Control, T_Stimulated
    groups = [
        ("B", "Control",    palette_B_ctrl, "B Control"),
        ("B", "Stimulated", palette_B_stim, "B Stimulated"),
        ("T", "Control",    palette_T_ctrl, "T Control"),
        ("T", "Stimulated", palette_T_stim, "T Stimulated"),
    ]
    
    for zone, condition, color, label in groups:
        values = np.concatenate([d[variable] for d in data
                                 if d["zone"] == zone and d["condition"] == condition])
        values = values[~np.isnan(values)]
        
        if len(values) > 0:
            all_values.append(values)
            all_labels.append(label)
            all_colors.append(color)
    
    plt.figure(figsize=(10, 6), dpi=150)
    
    # Violin plot
    sns.violinplot(
        data=all_values,
        inner="box",
        palette=all_colors,
        linewidth=1
    )
    
    # Stripplot
    sns.stripplot(
        data=all_values,
        jitter=True,
        color="black",
        size=3,
        alpha=0.5
    )
    
    # Calcul global pour les marges
    global_max = max([np.max(v) for v in all_values])
    global_min = min([np.min(v) for v in all_values])
    margin = 0.25 * (global_max - global_min)
    plt.ylim(global_min, global_max + margin)
    
    # Médiane
    for j, values in enumerate(all_values):
        median_val = np.median(values)
        all_median.append(median_val)
        plt.text(
            j,
            global_max + 0.1 * (global_max - global_min),
            f"{median_val:.3f}",
            ha='center',
            va='bottom',
            fontsize=8,
            color='red',
            fontweight='bold'
        )
    
    # Séparation B/T
    n_B = sum(1 for zone, condition, _, _ in groups
              if zone == "B" and any(
                  d["zone"] == zone and d["condition"] == condition
                  for d in data))
    plt.axvline(n_B - 0.5, color='black', linestyle='--')
    
    # Titres de zone
    plt.text(n_B / 2 - 0.5, global_max + 0.2 * (global_max - global_min),
             "Zone B", ha='center', fontsize=12, fontweight='bold')
    plt.text(n_B + (len(all_values) - n_B) / 2 - 0.5, global_max + 0.2 * (global_max - global_min),
             "Zone T", ha='center', fontsize=12, fontweight='bold')
    
    # Labels
    plt.xticks(
        ticks=np.arange(len(all_labels)),
        labels=all_labels,
        rotation=15,
        fontsize=9
    )
    
    plt.ylabel(ylabel)
    plt.title(f"Distribution de {variable} par zone et condition")
    plt.xlabel("")
    plt.tight_layout()
    plt.show()
    print("Médianes :", dict(zip(all_labels, all_median)))
    
    
    #%%
    
    
def plot_alpha2_condition_BT(data, variable="alpha2_filtered", ylabel="Alpha (no units)"):
    """
    Violin plot de alpha2 (ou autre variable) séparé par zone B/T et condition Control/Stimulated.
    """
    all_values = []
    all_labels = []
    all_colors = []
    all_median = []
    
    palette_B_ctrl = sns.color_palette("Purples", n_colors=5)[0]   # violet moyen
    palette_B_stim = sns.color_palette("Purples", n_colors=5)[2]   # violet foncé
    palette_T_ctrl = sns.color_palette("Greens",  n_colors=5)[0]   # vert moyen
    palette_T_stim = sns.color_palette("Greens",  n_colors=5)[2]   # vert foncé
    
    # Ordre : B_Control, B_Stimulated, T_Control, T_Stimulated
    groups = [
        ("B", "Control",    palette_B_ctrl, "B Control"),
        ("B", "Stimulated", palette_B_stim, "B Stimulated"),
        ("T", "Control",    palette_T_ctrl, "T Control"),
        ("T", "Stimulated", palette_T_stim, "T Stimulated"),
    ]
    
    for zone, condition, color, label in groups:
        values = np.concatenate([d[variable] for d in data
                                 if d["zone"] == zone and d["condition"] == condition])
        if len(values) > 0:
            all_values.append(values)
            all_labels.append(label)
            all_colors.append(color)
    
    plt.figure(figsize=(10, 6), dpi=150)
    
    # Violin plot
    sns.violinplot(
        data=all_values,
        inner="box",
        palette=all_colors,
        linewidth=1
    )
    
    # Stripplot
    sns.stripplot(
        data=all_values,
        jitter=True,
        color="black",
        size=3,
        alpha=0.5
    )
    
    # Calcul global pour les marges
    global_max = max([np.max(v) for v in all_values])
    global_min = min([np.min(v) for v in all_values])
    margin = 0.25 * (global_max - global_min)
    plt.ylim(global_min, global_max + margin)
    
    # Médiane
    for j, values in enumerate(all_values):
        median_val = np.median(values)
        all_median.append(median_val)
        plt.text(
            j,
            global_max + 0.1 * (global_max - global_min),
            f"{median_val:.3f}",
            ha='center',
            va='bottom',
            fontsize=8,
            color='red',
            fontweight='bold'
        )
    
    # Séparation B/T
    n_B = sum(1 for zone, condition, _, _ in groups
              if zone == "B" and any(
                  d["zone"] == zone and d["condition"] == condition
                  for d in data))
    plt.axvline(n_B - 0.5, color='black', linestyle='--')
    
    # Titres de zone
    plt.text(n_B / 2 - 0.5, global_max + 0.2 * (global_max - global_min),
             "Zone B", ha='center', fontsize=12, fontweight='bold')
    plt.text(n_B + (len(all_values) - n_B) / 2 - 0.5, global_max + 0.2 * (global_max - global_min),
             "Zone T", ha='center', fontsize=12, fontweight='bold')
    
    # Labels
    plt.xticks(
        ticks=np.arange(len(all_labels)),
        labels=all_labels,
        rotation=15,
        fontsize=9
    )
    
    plt.ylabel(ylabel)
    plt.title(f"Distribution de {variable} par zone et condition")
    plt.xlabel("")
    plt.tight_layout()
    plt.show()
    print("Médianes :", dict(zip(all_labels, all_median)))
    
    #%%Comparaison diffusion D, medium vs zone B et medium vs T zone 
    
def plot_D2_noLN_vs_zones(data, variable="D2_filtered", ylabel="D2 (µm²/s)"):
    """
    Deux violin plots côte à côte :
    - Gauche : No LN vs Zone B
    - Droite  : No LN vs Zone T
    """
    palette_noLN = sns.color_palette("Greys",   n_colors=5)[2]
    palette_B    = sns.color_palette("Purples", n_colors=5)[2]
    palette_T    = sns.color_palette("Greens",  n_colors=5)[2]
    
    # Récupération des valeurs
    values_noLN = np.concatenate([d[variable] for d in data if d["zone"] == "No LN"])
    values_B    = np.concatenate([d[variable] for d in data if d["zone"] == "B"])
    values_T    = np.concatenate([d[variable] for d in data if d["zone"] == "T"])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=150)
    
    for ax, values_zone, color_zone, zone_label in [
        (axes[0], values_B, palette_B, "Zone B"),
        (axes[1], values_T, palette_T, "Zone T"),
    ]:
        all_values = [values_noLN, values_zone]
        all_colors = [palette_noLN, color_zone]
        all_labels = ["No LN", zone_label]
        
        # Violin plot
        sns.violinplot(data=all_values, inner="box", palette=all_colors, linewidth=1, ax=ax)
        
        # Stripplot
        sns.stripplot(data=all_values, jitter=True, color="black", size=3, alpha=0.5, ax=ax)
        
        # Marges
        global_max = max([np.max(v) for v in all_values])
        global_min = min([np.min(v) for v in all_values])
        margin = 0.25 * (global_max - global_min)
        ax.set_ylim(global_min, global_max + margin)
        
        # Médiane
        for j, values in enumerate(all_values):
            median_val = np.median(values)
            ax.text(
                j,
                global_max + 0.1 * (global_max - global_min),
                f"{median_val:.3f}",
                ha='center',
                va='bottom',
                fontsize=8,
                color='red',
                fontweight='bold'
            )
        
        # Mann-Whitney
        _, p_value = mannwhitneyu(values_noLN, values_zone, alternative='two-sided')
        ax.set_title(f"No LN vs {zone_label}\np = {p_value:.2e}", fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(all_labels, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
    
    plt.suptitle(f"Distribution de {variable} : milieu vs ganglion", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    #%%Comparison alpha, medium vs B zone and medium vs T zone 
    
def plot_alpha2_noLN_vs_zones(data, variable="alpha2_filtered", ylabel="Alpha"):
    """
    Deux violin plots côte à côte :
    - Gauche : No LN vs Zone B
    - Droite  : No LN vs Zone T
    """
    palette_noLN = sns.color_palette("Greys",   n_colors=5)[2]
    palette_B    = sns.color_palette("Oranges", n_colors=5)[2]
    palette_T    = sns.color_palette("Blues",  n_colors=5)[2]
    
    # Récupération des valeurs
    values_noLN = np.concatenate([d[variable] for d in data if d["zone"] == "No LN"])
    values_B    = np.concatenate([d[variable] for d in data if d["zone"] == "B"])
    values_T    = np.concatenate([d[variable] for d in data if d["zone"] == "T"])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=150)
    
    for ax, values_zone, color_zone, zone_label in [
        (axes[0], values_B, palette_B, "Zone B"),
        (axes[1], values_T, palette_T, "Zone T"),
    ]:
        all_values = [values_noLN, values_zone]
        all_colors = [palette_noLN, color_zone]
        all_labels = ["No LN", zone_label]
        
        # Violin plot
        sns.violinplot(data=all_values, inner="box", palette=all_colors, linewidth=1, ax=ax)
        
        # Stripplot
        sns.stripplot(data=all_values, jitter=True, color="black", size=3, alpha=0.5, ax=ax)
        
        # Marges
        global_max = max([np.max(v) for v in all_values])
        global_min = min([np.min(v) for v in all_values])
        margin = 0.25 * (global_max - global_min)
        ax.set_ylim(global_min, global_max + margin)
        
        # Médiane
        for j, values in enumerate(all_values):
            median_val = np.median(values)
            ax.text(
                j,
                global_max + 0.1 * (global_max - global_min),
                f"{median_val:.3f}",
                ha='center',
                va='bottom',
                fontsize=8,
                color='red',
                fontweight='bold'
            )
        
        # Mann-Whitney
        _, p_value = mannwhitneyu(values_noLN, values_zone, alternative='two-sided')
        ax.set_title(f"No LN vs {zone_label}\np = {p_value:.2e}", fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(all_labels, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
    
    plt.suptitle(f"Distribution de {variable} : milieu vs ganglion", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()


#%% Comparison diffusion D : B zone vs T zone 

def plot_D2_B_vs_T(data, variable="D2_filtered", ylabel="D2 (µm²/s)"):
    """
    Violin plot : Zone B vs Zone T.
    """
    palette_B = sns.color_palette("Purples", n_colors=5)[2]
    palette_T = sns.color_palette("Greens",  n_colors=5)[2]
    
    values_B = np.concatenate([d[variable] for d in data if d["zone"] == "B"])
    values_T = np.concatenate([d[variable] for d in data if d["zone"] == "T"])
    
    all_values = [values_B, values_T]
    all_colors = [palette_B, palette_T]
    all_labels = ["Zone B", "Zone T"]
    
    plt.figure(figsize=(6, 6), dpi=150)
    
    sns.violinplot(data=all_values, inner="box", palette=all_colors, linewidth=1)
    sns.stripplot(data=all_values, jitter=True, color="black", size=3, alpha=0.5)
    
    global_max = max([np.max(v) for v in all_values])
    global_min = min([np.min(v) for v in all_values])
    margin = 0.25 * (global_max - global_min)
    plt.ylim(global_min, global_max + margin)
    
    # Médiane
    for j, values in enumerate(all_values):
        median_val = np.median(values)
        plt.text(
            j,
            global_max + 0.1 * (global_max - global_min),
            f"{median_val:.3f}",
            ha='center', va='bottom',
            fontsize=8, color='red', fontweight='bold'
        )
    
    # Mann-Whitney
    _, p_value = mannwhitneyu(values_B, values_T, alternative='two-sided')
    
    plt.xticks(ticks=[0, 1], labels=all_labels, fontsize=10)
    plt.ylabel(ylabel)
    plt.title(f"Zone B vs Zone T\np = {p_value:.2e}", fontsize=10)
    plt.tight_layout()
    plt.show()

#%%% Comparison alpha : B zone vs T zone 

def plot_alpha2_B_vs_T(data, variable="alpha2_filtered", ylabel="Alpha (no unit)"):
    """
    Violin plot : Zone B vs Zone T.
    """
    palette_B = sns.color_palette("Oranges", n_colors=5)[2]
    palette_T = sns.color_palette("Blues",  n_colors=5)[2]
    
    values_B = np.concatenate([d[variable] for d in data if d["zone"] == "B"])
    values_T = np.concatenate([d[variable] for d in data if d["zone"] == "T"])
    
    all_values = [values_B, values_T]
    all_colors = [palette_B, palette_T]
    all_labels = ["Zone B", "Zone T"]
    
    plt.figure(figsize=(6, 6), dpi=150)
    
    sns.violinplot(data=all_values, inner="box", palette=all_colors, linewidth=1)
    sns.stripplot(data=all_values, jitter=True, color="black", size=3, alpha=0.5)
    
    global_max = max([np.max(v) for v in all_values])
    global_min = min([np.min(v) for v in all_values])
    margin = 0.25 * (global_max - global_min)
    plt.ylim(global_min, global_max + margin)
    
    # Médiane
    for j, values in enumerate(all_values):
        median_val = np.median(values)
        plt.text(
            j,
            global_max + 0.1 * (global_max - global_min),
            f"{median_val:.3f}",
            ha='center', va='bottom',
            fontsize=8, color='red', fontweight='bold'
        )
    
    # Student test
    _, p_value = ttest_ind(values_B, values_T, alternative='two-sided')
    
    plt.xticks(ticks=[0, 1], labels=all_labels, fontsize=10)
    plt.ylabel(ylabel)
    plt.title(f"Zone B vs Zone T\np = {p_value:.2e}", fontsize=10)
    plt.tight_layout()
    plt.show()
    


#%% Comparison of the pore size between the B zone and T zone

def plot_pore_B_vs_T(data, variable = "pore_size_filtered", ylabel = "Pore size (um)"):
    
    """
    Representation of the distribution of pore size (um) : B zone vs T zone
    """
    
    palette_B = sns.color_palette("Reds", n_colors =5)[2]
    palette_T = sns.color_palette("Blues", n_colors = 5)[2]
    
    values_B = np.concatenate([d[variable] for d in data if d["zone"] == "B"])
    values_B = values_B[~np.isnan(values_B)]
    values_T = np.concatenate([d[variable] for d in data if d["zone"] == "T"])
    values_T = values_T[~np.isnan(values_T)]
    
    all_values = [values_B, values_T]
    all_colors = [palette_B, palette_T]
    all_labels = ["Zone B", "Zone T"]
  
    plt.figure(figsize=(6, 6), dpi=150)
  
    sns.violinplot(data=all_values, inner="box", palette=all_colors, linewidth=1)
    sns.stripplot(data=all_values, jitter=True, color="black", size=3, alpha=0.5)
  
    global_max = max([np.max(v) for v in all_values])
    global_min = min([np.min(v) for v in all_values])
    margin = 0.25 * (global_max - global_min)
    plt.ylim(global_min, global_max + margin)
  
  # Médiane
    for j, values in enumerate(all_values):
      median_val = np.median(values)
      plt.text(
          j,
          global_max + 0.1 * (global_max - global_min),
          f"{median_val:.3f}",
          ha='center', va='bottom',
          fontsize=8, color='red', fontweight='bold'
      )
  
  # Student test
    _, p_value = ttest_ind(values_B, values_T, alternative='two-sided')
  
    plt.xticks(ticks=[0, 1], labels=all_labels, fontsize=10)
    plt.ylabel(ylabel)
    plt.title(f"Zone B vs Zone T\np = {p_value:.2e}", fontsize=10)
    plt.tight_layout()
    plt.show()

#%% PORE SIZE between CTRL and STIM 

def pore_ctrlstim(data, variable="pore_size_filtered", ylabel="Pore size (µm)"):
    """
    Representation of the distribution of pore size (µm) : Control vs Stimulated
    """
    palette_ctrl = sns.color_palette("Oranges", n_colors=5)[2]
    palette_stim = sns.color_palette("Reds",    n_colors=5)[2]
    
    values_ctrl = np.concatenate([d[variable] for d in data if d["condition"] == "Control"])
    values_ctrl = values_ctrl[~np.isnan(values_ctrl)]
    values_stim = np.concatenate([d[variable] for d in data if d["condition"] == "Stimulated"])
    values_stim = values_stim[~np.isnan(values_stim)]
    
    all_values = [values_ctrl, values_stim]
    all_colors = [palette_ctrl, palette_stim]
    all_labels = ["Control", "Stimulated"]
    
    plt.figure(figsize=(6, 6), dpi=150)
    
    sns.violinplot(data=all_values, inner="box", palette=all_colors, linewidth=1)
    sns.stripplot(data=all_values, jitter=True, color="black", size=3, alpha=0.5)
    
    global_max = max([np.max(v) for v in all_values])
    global_min = min([np.min(v) for v in all_values])
    margin = 0.25 * (global_max - global_min)
    plt.ylim(global_min, global_max + margin)
    
    # Médiane
    for j, values in enumerate(all_values):
        median_val = np.nanmedian(values)
        plt.text(
            j,
            global_max + 0.1 * (global_max - global_min),
            f"{median_val:.3f}",
            ha='center', va='bottom',
            fontsize=8, color='red', fontweight='bold'
        )
    
    # Choix du test selon normalité
    _, p_shapiro_ctrl = shapiro(values_ctrl)
    _, p_shapiro_stim = shapiro(values_stim)
    both_normal = p_shapiro_ctrl > 0.05 and p_shapiro_stim > 0.05
    
    if both_normal:
        _, p_value = ttest_ind(values_ctrl, values_stim, alternative='two-sided')
        test_name = "Student"
    else:
        _, p_value = mannwhitneyu(values_ctrl, values_stim, alternative='two-sided')
        test_name = "Mann-Whitney"
    
    plt.xticks(ticks=[0, 1], labels=all_labels, fontsize=10)
    plt.ylabel(ylabel)
    plt.title(f"Control vs Stimulated ({test_name})\np = {p_value:.2e}", fontsize=10)
    plt.tight_layout()
    plt.show()


# Util


