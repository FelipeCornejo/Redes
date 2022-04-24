# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 23:35:45 2021

@author: littl
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft,fftfreq,ifft

def plotter(figura, titulo, xlab, ylab, x, y, color="blue",frecuencia=1,ylim=0):
    #plt.figure(nombre, tamaño, resolución)
    plt.figure(figura)
    plt.title(titulo)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if ylim != 0:
        plt.ylim(ylim)
    if frecuencia != 1:
        x_train = np.arange(-1,1,frecuencia)
        y_train = np.sinc(5*x_train)**2
        plt.stem(x_train,y_train)
        plt.scatter(x_train,y_train)
    plt.plot(x, y, color,linewidth=1,color=color)
    plt.show()

def espectro_plotter(figura, titulo, xlab, ylab, x, y):
    #plt.figure(nombre, tamaño, resolución)
    plt.figure(figura)
    plt.title(titulo)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.specgram(x,Fs=y)
    plt.show()

def create_train(x,y,f):
    i = 0
    limx = len(x)
    print(limx)
    ympulso = []
    freq = 0
    while i < limx:
        if freq == (limx/f):
            freq = 0
            ympulso.append(1)
        else:
            ympulso.append(0)
        freq +=1
        i+= 1
    ympulso = np.array(ympulso)
    ympulso = ympulso * y
    return ympulso
    
# Pregunta 1. (3 pts.) Una señal g(t)=sinc²(5t) se muestrea usando un tren de impulsos uniformemente espaciados a una frecuencia 
# de 5 Hz, 10 Hz y 20 Hz. Para cada una de las tres frecuencias responda:
# a) Grafique la señal muestreada

# g(t) corresponde a sin(pi*x)/(pi*x)
x = np.arange(-1,1,0.001)
y = np.sinc(5*x)**2

f5 = 5
f10 = 10
f20 = 20

ympulso5 = create_train(x,y,f5*2)
ympulso10 = create_train(x,y,f10*2)
ympulso20 = create_train(x,y,f20*2)

plotter("Figura a","g(x) con T.I. de 5 Hz","Tiempo","Amp",x,y,frecuencia=1/f5)
plotter("Figura a","g(x) con T.I. de 10 Hz","Tiempo","Amp",x,y,frecuencia=1/f10)
plotter("Figura a","g(x) con T.I. de 20 Hz","Tiempo","Amp",x,y,frecuencia=1/f20)

plotter("figura", "titulo", "xlab", "ylab", x, ympulso5)
plotter("figura", "titulo", "xlab", "ylab", x, ympulso10)
plotter("figura", "titulo", "xlab", "ylab", x, ympulso20)

# (0.9) Grafique la transformada de Fourier de la señal muestreada

y_fourier5 = fft(ympulso5)
y_fourier10 = fft(ympulso10)
y_fourier20 = fft(ympulso20)
x_fourier_freq5 = fftfreq(y_fourier5)
x_fourier_freq10 = fftfreq(y_fourier10)
x_fourier_freq20 = fftfreq(y_fourier20)

plotter("figura", "titulo", "xlab", "ylab", x_fourier_freq5, abs(y_fourier5))
plotter("figura", "titulo", "xlab", "ylab", x_fourier_freq10, abs(y_fourier10))
plotter("figura", "titulo", "xlab", "ylab", x_fourier_freq20, abs(y_fourier20))




