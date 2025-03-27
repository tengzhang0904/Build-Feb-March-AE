# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 23:26:37 2025

@author: TengZhang
"""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# Define parameters
V = 2.0  # Coupling strength
alpha = 1.0  # Energy sweep rate (dE/dt)
t_range = np.linspace(-10, 10, 10000)  # Time evolution range

dt = t_range[1] - t_range[0]  # Time step
hbar = 1.0  # Reduced Planck’s constant

# Initial state (starting in state |1>)
psi = np.array([1.0, 0.0], dtype=complex)

# Time evolution using the Schrödinger equation
psi_t = []
es_t = []
for t in t_range:
    H = np.array([[alpha * t, V], [V, -alpha * t]])  # Hamiltonian
    U = la.expm(-1j * H * dt / hbar)  # Time evolution operator
    psi = U @ psi  # Apply time evolution
    eigenvalues, eigenvectors = la.eig(H)
    
    psi_t.append(psi)
    es_t.append(np.sort(eigenvalues.real))

# Convert to array
psi_t = np.array(psi_t)

# Compute probabilities
P1 = np.abs(psi_t[:, 0])**2  # Probability of staying in initial state
P2 = np.abs(psi_t[:, 1])**2  # Probability of transition

# Analytical Landau-Zener probability
P_LZ = np.exp(-2 * np.pi * V**2 / (hbar * alpha))

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(t_range, P1, label="State |1> Probability")
plt.plot(t_range, P2, label="State |2> Probability")
plt.axhline(y=P_LZ, color='r', linestyle="--", label=f"LZ Prediction: {P_LZ:.3f}")
plt.xlabel("Time")
plt.ylabel("Probability")
plt.legend()
plt.title("Landau-Zener Transition")
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(t_range, np.array(es_t))
plt.plot(t_range, t_range,'--')
plt.plot(t_range, -t_range,'--')

plt.xlabel("Time")
plt.ylabel("Energy")
plt.title("Landau-Zener Transition")
plt.grid()
plt.show()