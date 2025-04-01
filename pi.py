# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 16:19:26 2025

@author: TengZhang
"""

import random

def monte_carlo_pi(num_samples: int) -> float:
    inside_circle = 0
    
    for _ in range(num_samples):
        x, y = random.uniform(-1, 1), random.uniform(-1, 1)
        if x**2 + y**2 <= 1:
            inside_circle += 1
    
    return (inside_circle / num_samples) * 4


num_samples = 1000  # Number of random points
pi_estimate = monte_carlo_pi(num_samples)
print(f"Estimated π value: {pi_estimate}")