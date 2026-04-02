#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 12:30:57 2026

@author: xuan_nguyen
"""
from scipy.integrate import quad
import numpy as np;
import matplotlib.pyplot as plt;
from scipy.interpolate import interp1d # Import interp1d
    


"""    
def numerical_LT(msd, dt, omega_range):
  Computes numerical Laplace transform of the MSD
  
  Parameters : 
      msd : array of the mean MSD 2D 
      dt : no time (????)
      omega_range : range of the angular frequencies 
      
  Returns : 
      G1 : elastic/storage modulus (Pa)
      G2 : viscous/loss modulus (Pa)
      
  kB = 1.380649e-23
  T = 310 
  a = 0.255e-6
  
  msd = msd[0:160]
  N = len(msd)
  t = dt * np.arange(1, N+1) #avoids 0 
  t = t[0:160]

  mid = N//2 #WHY ?
  freqs = np.fft.fftfreq(N, dt)
  omega = 2 * np.pi * freqs
  omega = omega[1:mid]

  G1 = np.zeros_like(omega_range)
  G2 = np.zeros_like(omega_range)
  
  for k, w in enumerate(omega): #for every enumeration, index k is associated 
  #interpolate mean MSD function
  #create function as object (integrand_real) with variable t_Val 
  
        integrand_real = lambda t_val: np.interp(t_val, t, msd) * np.exp(-w * t_val) #Laplace nucleus
        msd_LT, _ = quad(integrand_real, t[0], t[-1]) #Laplace transform 
        G_star = (kB * T) / (np.pi * a * msd_LT)
    
return omega """


def numerical_LT_TEST(msd, dt, omega_range):
    """
    Computes numerical Laplace transform of the MSD.
    
    Parameters : 
        msd         : array of the mean MSD 2D (m²)
        dt          : time step (s)
        omega_range : array of angular frequencies (rad/s)
      
    Returns : 
        omega_range : angular frequencies (rad/s)
        G1          : elastic/storage modulus (Pa)
        G2          : viscous/loss modulus (Pa)
    """
    kB = 1.380649e-23
    T  = 310 
    a  = 0.255e-6
    
    N = len(msd)
    t = dt * np.arange(1, N + 1)  # évite t=0
    
    G1 = np.zeros_like(omega_range)
    G2 = np.zeros_like(omega_range)
    
    for k, w in enumerate(omega_range):  # itère sur omega_range et non omega
        s = 1j * w
        
        # Noyau de Laplace e^(-iωt) séparé en parties réelle et imaginaire
        real_part, _ = quad(lambda t_val: np.interp(t_val, t, msd) * np.cos(w * t_val), t[0], t[-1])
        imag_part, _ = quad(lambda t_val: np.interp(t_val, t, msd) * np.sin(w * t_val), t[0], t[-1])
        
        msd_LT = real_part - 1j * imag_part  # transformée complexe
        
        # GSER
        G_star = (kB * T) / (np.pi * a * s * msd_LT)
        
        G1[k] = np.real(G_star)
        G2[k] = np.imag(G_star)
    
    return omega_range, G1, G2
    



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
