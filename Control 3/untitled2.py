# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 22:08:32 2021

@author: littl
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft,fftfreq,ifft

x = np.linspace(-3, 3, 300)
y = np.sinc(5*np.pi*x)
plt.subplot(2,1,1)
plt.plot(x,y)
plt.show()
plt.grid(True)
plt.title("Señal sinc(5*pi*x)")
plt.subplot(2,1,2)
FS_5=5
FD=1/5
n = np.linspace(-3,3,5)
xn = np.sinc(5*np.pi*n)
plt.stem(n,xn)
plt.title("Señal muestreada a 5 Hz")