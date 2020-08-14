# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 17:08:56 2020

@author: benno
"""


import matplotlib.pyplot as plt
from scipy.stats import linregress

plt.figure()
plt.plot([0.05, 0.025, 0.015, 0.008], [0.88, 1.26, 1.4, 1.47], '.-')
plt.xlim(0, 0.055)
plt.ylim(0, 2)


f = linregress([0.05, 0.025, 0.015, 0.008], [0.88, 1.26, 1.4, 1.47])
print(f)
