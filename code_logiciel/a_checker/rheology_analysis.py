#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to compute G', G'' and confinament radius from SPT experiments

Created on Mon Mar  9 16:23:37 2026

@author: Paolo Pierobon paolo.pierobon@inserm.fr
"""



import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
from scipy.fft import fft, fftfreq
import sympy as sym

#%% 


def complex_modulusFFT(msd, dt, a):
    """
    Compute G′ and G″ from MSD using GSER.

    Parameters:
    - msd: array of mean squared displacement vs. lag time
    - dt: time step between trajectory points
    - a: particle radius (m)
    - T: temperature (K)

    Returns:
    - freqs: frequency array
    - G_prime: storage modulus
    - G_double_prime: loss modulus
    
    Use: 
        
    omega, G1, G2 = complex_modulusFFT(msd, dt, a):
    plt.plot(omega, G1)
    
    
    """
    kB = 1.380649e-23  # Boltzmann constant
    T = 310 # 37 celsius
    N = len(msd)


    # # One-sided FT of MSD: integrate from 0 to inf
    # # rfft gives the non-negative frequency components
    msd_ft = np.fft.rfft(msd) * dt        # one-sided FT (scaled by dt)
    freqs  = np.fft.rfftfreq(N, d=dt)

    # Avoid division by zero at DC (freq=0)
    omega = 2 * np.pi * freqs[1:]
    msd_ft = msd_ft[1:]

    # GSER in Fourier space: G*(ω) = kT / (πa · iω · F(MSD)(ω))
    
    Gstar = kB * T / (np.pi * a * 1j * omega * msd_ft)

    G1 = Gstar.real   # storage modulus
    G2 = Gstar.imag   # loss modulus
    return omega, G1, G2

#%% 



# Kramer Kronig control
from scipy.integrate import quad
def KK_check(omega_0, omega_arr, G_doubleprime_arr):
    """
    Reconstruct G'(ω₀) from G''(ω) via Kramers-Kronig.
    If reconstructed G' matches measured G', your data is consistent
    with a causal passive-like response.
    """
    from scipy.interpolate import interp1d
    Gpp_interp = interp1d(omega_arr, G_doubleprime_arr, 
                          bounds_error=False, fill_value=0)
    
    integrand = lambda w: 2*w * Gpp_interp(w) / (w**2 - omega_0**2 + 1e-30)
    result, _ = quad(integrand, omega_arr[0], omega_arr[-1])
    return (2 * omega_0 / np.pi) * result  # reconstructed G'


def KK_check_PV(omega_0, omega_arr, G_doubleprime_arr):
    from scipy.interpolate import interp1d
    Gpp = interp1d(omega_arr, G_doubleprime_arr, 
                   bounds_error=False, fill_value=0)
    
    eps = omega_0 * 0.01  # small exclusion window around singularity
    
    integrand = lambda w: 2*w * Gpp(w) / (w**2 - omega_0**2)
    
    r1, _ = quad(integrand, omega_arr[0],   omega_0 - eps)
    r2, _ = quad(integrand, omega_0 + eps,  omega_arr[-1])   
    return (2 * omega_0 / np.pi) * (r1 + r2)

# for traj in tracks.values():
#     msd=traj['MSD2D']
#     dt=traj['dt']
#     om, G1, G2 = complex_modulusFFT(msd, dt, 0.25)
#     traj['om']=om
#     traj['G1']=G1
#     traj['G2']=G2
#     print(traj['id'])


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
        poly_sym += c * tau**(5-i)

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


def confinement_radius(msd, dt, a):
    alphatime, Dtime = MSDtime(msd, dt,10)
    idx=np.where(alphatime<0.1)[0] # to be checked
    if idx.size>0:
        idx0=idx[0]
        R=np.sqrt(np.nanmean(msd[idx0:]))+a
    else:
        R=np.nan
    return R