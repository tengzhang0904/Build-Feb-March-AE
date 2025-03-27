# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 11:06:53 2025

@author: TengZhang
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the Rabi oscillation Hamiltonian
def rabi_oscillation(t, c, Omega):
    H = np.array([[0, Omega / 2], [Omega / 2, 0]])
    return -1j * np.dot(H, c)

# Parameters
Omega = 1.0  # Rabi frequency in arbitrary units
t_max = 10    # Maximum time
num_points = 1000  # Number of time points

# Initial state: |psi(0)> = |0>
c0 = np.array([1, 0], dtype=complex)

# Time span
t_eval = np.linspace(0, t_max, num_points)

# Solve the Schrödinger equation
sol = solve_ivp(rabi_oscillation, [0, t_max], c0, t_eval=t_eval, args=(Omega,))

# Extract probabilities
P_0 = np.abs(sol.y[0])**2  # Probability of being in state |0>
P_1 = np.abs(sol.y[1])**2  # Probability of being in state |1>

# Plot the results
plt.figure(figsize=(8, 5))
plt.plot(sol.t, P_0, label='$P_0$', lw=2)
plt.plot(sol.t, P_1, label='$P_1$', lw=2, linestyle='dashed')
plt.xlabel('Time')
plt.ylabel('Probability')
plt.title('Rabi Oscillations in a Two-Level System')
plt.legend()
plt.grid()
plt.show()
