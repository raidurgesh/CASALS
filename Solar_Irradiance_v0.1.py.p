#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 12:57:57 2025

@author: drai
"""

import numpy as np

# Parameters
E_inc = 1361.0           # Solar irradiance [W/m^2] at normal incidence (TOA)
rho   = 0.90             # Lambertian reflectance (albedo)
tel_dia = 0.6            # in m
Alt     = 500e3          # Altitude of spacecraft [m]
lambda_nm = 1040         # Wavelength for photon-rate estimate [nm]
Counting_wind = 100e-9   # Time window for counting / accumulating background [s]

# Constants
h = 6.62607015e-34       # Planck constant [J s]
c = 2.99792458e8         # Speed of light [m/s]

# Calculations

# Radiance of a Lambertian surface: L = E_inc * rho / pi
L = (E_inc * rho) / np.pi  # [W m^-2 sr^-1]

# Area of telescope
A_det = np.pi*(tel_dia/2)**2   # Detector (or lens) area [m^2] (in reality there should be some margin for M2 geometry)

# Solid angle subtended by telescope of area A_det at distance Alt
Omega = A_det / Alt**2       # [sr]

# Power reaching detector
P_det = L * A_det * Omega  # [W]

# Photon energy
lambda_m = lambda_nm * 1e-9
E_photon = (h * c) / lambda_m  # [J]

# Photon rate
photon_rate = P_det / E_photon  # [photons/s]

# Number of photons in one counting window
photon_wind = photon_rate * Counting_wind

# Output
print("-- Reflection Illumination Result --")
print(f"Incident irradiance      : {E_inc:.2f} W/m^2")
print(f"Surface reflectance (ρ)  : {rho:.2f}")
print(f"Radiance (L)             : {L:.3f} W/(m²·sr)")
print(f"Detector area            : {A_det:.4f} m²")
print(f"Distance to surface      : {Alt:.2f} m")
print(f"Solid angle (Ω)          : {Omega:.3e} sr")
print(f"Power on detector        : {P_det:.3e} W")
print(f"Photon wavelength        : {lambda_nm} nm")
print(f"Photon energy            : {E_photon:.3e} J")
print(f"Photon rate              : {photon_rate:.3e} photons/s")
print(f"Number of Photons        : {photon_wind:.3} photons within {Counting_wind*1e9} ns window")
