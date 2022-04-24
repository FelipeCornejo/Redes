# -*- coding: utf-8 -*-
'''
LABORATORIO 4: MODULACIÓN DIGITAL
Aylin Rodrí­guez
Jennifer Velozo
'''


import matplotlib.pyplot as plt
import numpy as np
import random as rn
from numpy import cos, pi

######################## FUNCIONES ########################

#Entradas: nada
#Salidas: un arreglo de largo 20 el cual contiene enteros 0 y 1 almacenados de forma aleatoria
#Genera 1 arreglo representando señales de bits aleatorios.
def gen_signal():
    array = []
    
    i = 0
    while i<20:
        array.append(rn.randint(0, 1))
        i+=1
    return array

#Entradas: la señal original y la tasa de bits.
#Salidas: np.array el cual funcionará como arreglo de tiempo.
#Genera 1 np.array que representa el tiempo de una función.  
def gen_time(signal,bitrate):
    t = np.linspace(0,len(signal)/bitrate,len(signal))
    return t

#Entrada: Señal a aplicar el ruido, entero que representa el SNR
#Salida: Señal con ruido blanco gaussiano aplicado
#Función que genera un ruido y lo aplica a la señal ingresada por parametro.
def gen_ruido(signal_mod, snr):
    ruido = np.random.normal(0, 1, len(signal_mod))
    energia_s = np.sum(np.abs(signal_mod) * np.abs(signal_mod))
    energia_n = np.sum(np.abs(ruido) * np.abs(signal_mod))
    snr_lineal = np.exp(snr/10)
    sigma = np.sqrt(energia_s / (energia_n * snr_lineal))
    ruido = sigma * ruido
    awgn = signal_mod + ruido
    return awgn

#Entrada: Nombre de la figura, Titulo a mostrar, label x, labely, vector x, vector y,  opcional : color de la linea a graficar
#Salida: muestra un grafico dependiendo de X y resultado Y
#Funcion que genera un grafico respecto a los parametros entregados
def plotter(figura, titulo, xlab, ylab, x, y,xlim=False,ylim=False,color="lightgreen"):
    #plt.figure(figura, (8,4))
    plt.figure(figura)
    plt.title(titulo)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if ylim != False:
        plt.ylim(ylim[0], ylim[1])
    if xlim != False:
        plt.xlim(xlim[0], xlim[1])
    plt.plot(x, y, color, linewidth = 0.7)
    #plt.savefig(figura)
    plt.show()
    

#Entrada: La señal a modular, la tasa de bits
#Salida: el tiempo de la señal modulada, La señal modulada en FSK, el largo del arreglo modulado.
#Modula una lista de 0 y 1 que represnenta una señal de datos binarios.
def FSK(signal, freq):
    largo = int(1/freq * 10 * freq * 2) # Esto es simplemente 20.
    tiempo = np.linspace(0, 1/freq,largo)
    
    signal_mod = []
    cero = np.cos(2 * np.pi * freq * tiempo)
    uno = np.cos(2 * np.pi * freq * 2 * tiempo) # EL doble de frecuencia.
    for bit in signal:
        if bit == 0:
            signal_mod.extend(cero)
        else:
            signal_mod.extend(uno)

    t2 = len(signal) * 1/freq  
    t_mod = np.linspace(0, t2, len(signal_mod))
    
    return t_mod, signal_mod, largo




#Entrada: tiempo de modulación, señal a demodular, el largo del arreglo modulado, la frecuencia o tasa de bits.
#Salida: La señal demodulada
#Función encargada de realizar demodulaciónn FSK
def DFSK(tiempo, signal, largobit, freq):
    cos_signal = np.cos(2 * np.pi * freq*2 * tiempo)
    demod = signal * cos_signal
    n_bits = int(len(demod) / largobit)

    signal_demod = [] 
    for i in range(1, n_bits + 1):
        voltaje = demod[((i - 1) * largobit): i * largobit - 1]
        mean_voltaje = np.mean(voltaje)
        
        if mean_voltaje > 0.25:
            signal_demod.append(1)
        else:
            signal_demod.append(0)
    return signal_demod

#Función que calcula la tasa de error binario en la señal demodulada.
#Entrada: señal demodulada y señal
#Salida: tasa de error
def get_error(bitsdemod, bits):
    contador = 0
    for i in range(len(bits)):
        if bits[i] != bitsdemod[i]:
            contador += 1
    ber = float(contador / len(bits))
    return ber


####################
##BLOQUE PRINCIPAL##
####################

### IMPLEMENTACIONES ### 
rn.seed(9)
signal = gen_signal()
freq = 100
t = gen_time(signal,freq)

#############
##GRAFICO 1##
#############
plotter("Figura1","Señal original", "Tiempo", "Amplitud", t, signal)

t_signalmod, signal_mod, largobit = FSK(signal,freq)
#t_mod = np.linspace(0, 1/freq  , len(signal))
#t_mod = np.linspace(0,len(signal_mod)/freq, len(signal_mod))

#############
##GRAFICO 2##
#############
plotter("Figura2", "Señal Modulada FSK", "Tiempo", "Amplitud", t_signalmod, signal_mod)

awgn = gen_ruido(signal_mod, 100)

#############
##GRAFICO 3##
#############
plotter("Figura3", "Ruido con SNR = 100","Tiempo", "Amplitud", t_signalmod, awgn)

signal_demod = DFSK(t_signalmod, signal_mod, largobit, freq)
t_demod = np.linspace(0, len(signal_demod)/freq, len(signal_demod))

#############
##GRAFICO 4##
#############

plotter("Figura4", "Señal demodulada","Tiempo", "Amplitud", t_demod, signal_demod)
errorBits = get_error(signal_demod, signal)

### SIMULACION DE CANAL ###
freq_sim = 500
L = 1e6
signal_sim = np.random.randint(2, size = int(L))
colores = ['-b', '-g', '-r']

plt.figure(1)
i = 0
while i < 3:
    snr_cum = []
    error_cum = []
    freq_sim = freq_sim + i*500 
    tiempo, signal_sim_mod, len_bit = FSK(signal_sim, freq_sim)
    snr = -2
    while snr < 11:
        
        awgn = gen_ruido(signal_sim_mod, snr)
        signal_sim_demod = DFSK(tiempo, awgn, len_bit, freq_sim) 
        ber = get_error(signal_sim_demod, signal_sim)
        snr_cum.append(snr)
        error_cum.append(ber)
        lab = str(freq_sim) + ' [bps]'
        snr += 1
        
    plt.plot(snr_cum, error_cum, colores[i], label=lab, linewidth=3, marker='x')
    i += 1

plt.grid(True)
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.yscale('log')
plt.xscale('linear')
plt.title('Rendimiento del sistema de comunicación')
plt.legend()
plt.show()
