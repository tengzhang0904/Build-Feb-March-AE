# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 15:59:49 2025

@author: TengZhang
"""

import numpy as np
import matplotlib.pyplot as plt

# Poor PRNG: Linear Congruential Generator (LCG) with bad parameters

A = 65539
M = 2**8
def bad_prng(seed, a=A, c=12345, m=M):
    while True:
        seed = (a * seed + c) % m
        yield seed / m  # Normalize to range [0,1]

# Generate pseudo-random points
seed = 42
prng = bad_prng(seed)
n = 1000
x = [next(prng) for _ in range(n)]
y = [next(prng) for _ in range(n)]

# Plot the pseudo-random points
plt.figure(figsize=(6,6))
plt.scatter(x, y, s=1, color='red')
plt.title("Poor PRNG - Visible Patterns in Randomness")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.show()