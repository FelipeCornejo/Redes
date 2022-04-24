# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 20:49:29 2021

@author: littl
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft,fftfreq,ifft

def plotter(figura, titulo, xlab, ylab, x, y, color="blue",frecuencia=1,ylim=0):
    #plt.figure(nombre, tamaño, resolución)
    plt.figure(figura)
    if ylim != 0:
        plt.ylim(ylim)
    if frecuencia != 1:
        plt.subplot(2, 1,2)
        plt.stem(x,y)
        plt.scatter(x,y)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
    plt.subplot(2, 1,1)
    plt.title(titulo)
    plt.ylabel(ylab)
    plt.plot(x, y, color,linewidth=1,color=color)
    plt.show()
    
# (0.6) Grafique la señal muestreada
# Para la señal muestrando con una frecuencia de 5 Hz
freq1 = 5
T1 = 1 / freq1
x1 = np.arange(-3,3, T1)
xT1 = x1 * T1
y1 = np.sinc(5 * np.pi * xT1) * np.sinc(5 * np.pi * xT1)

# Para la señal muestrando con una frecuencia de 10 Hz
freq2 = 10
T2 = 1 / freq2
x2 = np.arange(-3,3, T2)
xT2 = x2 * T2
y2 = np.sinc(5 * np.pi * xT2) * np.sinc(5 * np.pi* xT2)

# Para la señal muestrando con una frecuencia de 20 Hz

freq3 = 20
T3 = 1 / freq3
x3 = np.arange(-3, 3, T3)
xT3 = x3 * T3
y3 = np.sinc(5 * np.pi * xT3) * np.sinc(5 * np.pi * xT3)

plotter("a","Señal muestreada a 5 Hz", "tiempo (s)", "Amplitud", x1,y1,frecuencia=T1)
plotter("a","Señal muestreada a 10 Hz", "tiempo (s)", "Amplitud", x2,y2,frecuencia=T2)
plotter("a","Señal muestreada a 20 Hz", "tiempo (s)", "Amplitud", x3,y3,frecuencia=T3)

# (0.9) Grafique la transformada de Fourier de la señal muestreada

fft1 = fft(y1).real
ffreq1 = fftfreq(len(xT1))

plotter("a","Transformada de Señal muestreada a 5 Hz", "Frecuencia (Hz)", "|F(w)|", ffreq1,abs(fft1),frecuencia=T1)

fft2 = fft(y2).real
ffreq2 = fftfreq(len(xT2))

plotter("a","Transformada de Señal muestreada a 10 Hz", "Frecuencia (Hz)", "|F(w)|", ffreq2,abs(fft2),frecuencia=T2)

fft3 = fft(y3).real
ffreq3 = fftfreq(len(y3))

plotter("a","Transformada de Señal muestreada a 20 Hz", "Frecuencia (Hz)", "|F(w)|", ffreq3,abs(fft3),frecuencia=T3)




