import matplotlib.pyplot as plt
import numpy as np
import random as rn
from scipy.fftpack import fftshift
from numpy import cos, pi


#Entradas: nada
#Salidas: un arreglo de largo 10 y otro 100, los cuales contienen enteros 0 y 1 almacenados de forma aleatoria
#Genera 2 arreglos representando señales de bits aleatorios.
def gen_signals():
    array_10 = []
    array_100 = []

    i = 0
    while i<10:
        array_10.append(rn.randint(0, 1))
        i+=1
        
    i = 0
    while i<100:
        array_100.append(rn.randint(0, 1))
        i+=1
    return array_10, array_100

#Entradas: la tasa de bits.
#Salidas: np.array el cual funcionará como arreglo de tiempo.
#Genera 2 np.array que representan el tiempo de una función.  
def gen_time(tbits):
    t_10 = np.linspace(0, tbits,1000)
    return t_10, t_10

#Entrada: 2 np.array que representen el tiempo
#Salida: 2 Señales portadoras, la primera a 75 Hz y la segunda a 125 Hz
#Genera 2 señales portadoras.
def gen_porta(t_10,t_100):
    #Frecuencias
    freq0 = 75
    freq1 = 125 
    
    #Definición de las señales portadoras
    porta_75 = cos(2*pi*freq0*t_10)
    porta_125 = cos(2*pi*freq1*t_100)
    return porta_75, porta_125

#Entrada: La señal a modular, la señal portadora 1 y la señal portadora 2
#Salida: La señal modulada en FSK
#Modula una lista de 0 y 1 que represnenta una señal de datos binarios.
def modulador_FSK(signal,porta_75,porta_125):
    modulada = []
    for e in signal:
        if e == 0:
            modulada.extend(porta_75)
        else:
            modulada.extend(porta_125)
    return modulada

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
    
#Entrada:Arreglo de amplitudes de una señal, la frecuencia de la señal.
#Salida: Arreglo de amplitudes de la transformada de fourier, arreglo de frecuencias de la señal
#Funcion que calcula la transformada de fourier entregando el eje x e y (se hizo esta copia ya que la primera no dejaba ingresar por parametro datos de tipo np.ndarray)
def fourier(data,freq):
    largo = len(data)
    tiempo = largo/freq
    fourier = np.fft.fft(data)
    aux = np.arange(-len(fourier)/2,len(fourier)/2)
    #shift
    ffreq=fftshift(aux/tiempo)
    return fourier,ffreq

#Definir Tasa de bits.
tbits = 1/10

#Generar las señales correspondientes a 10 y 100 bits.
array_10, array_100 = gen_signals()
#Generar los vectores de tiempo debido a una tasa de 10 bits por segundo
t_10, t_100 = gen_time(tbits)
#Generar las señales portadoras debido a los vectores de tiempo.
porta_75, porta_125 = gen_porta(t_10,t_100)

#Modulacion FSK
modulada_10 = modulador_FSK(array_10, porta_75, porta_125)
modulada_100 = modulador_FSK(array_100, porta_75, porta_125)

tiempo_10 = np.linspace(0,len(array_10)*tbits,1000*len(array_10))
tiempo_100 = np.linspace(0,len(array_100)*tbits,len(array_100))

plotter("figura1", "Señal 10 bits vs tiempo", "Tiempo [s]", "Amplitud", tiempo_10, modulada_10)

fourier_100,ffreq_100 = fourier(modulada_100,125)

plotter("figura2", "Señal 100 bits vs frecuencia", "Frecuencia [Hz]", "Amplitud", ffreq_100, abs(fourier_100))

#Definir Tasa de bits.
tbits_20 = 20
tbits_3 = 3
tbits_2 = 2
tbits_1 = 1
tbits_100 = 1/50
tbits_1000 = 1/100
tbits_10000 = 1/200
tbits_12000 = 1/250
tbits_12500 = 1/300
tbits_20000 = 1/500
tbits_20300 = 1/1000

#Para tbits_100
t_10, t_100 = gen_time(tbits_20)
porta_75, porta_125 = gen_porta(t_10,t_100)
#Modulacion FSK
modulada_100 = modulador_FSK(array_100, porta_75, porta_125)

fourier_100,ffreq_100 = fourier(modulada_100,75)

plotter("figura2.5", "Señal 100 bits", "Tiempo ", "Amplitud", tiempo_100, array_100)

plotter("figura2.5", "Señal 100 bits vs frecuencia Tbits = 1", "Frecuencia [Hz]", "Amplitud", ffreq_100, abs(fourier_100))

#Para tbits_100
t_10, t_100 = gen_time(tbits_3)
porta_75, porta_125 = gen_porta(t_10,t_100)
#Modulacion FSK
modulada_100 = modulador_FSK(array_100, porta_75, porta_125)

fourier_100,ffreq_100 = fourier(modulada_100,75)

plotter("figura2.5", "Señal 100 bits vs frecuencia Tbits = 1", "Frecuencia [Hz]", "Amplitud", ffreq_100, abs(fourier_100))

#Para tbits_100
t_10, t_100 = gen_time(tbits_2)
porta_75, porta_125 = gen_porta(t_10,t_100)
#Modulacion FSK
modulada_100 = modulador_FSK(array_100, porta_75, porta_125)

fourier_100,ffreq_100 = fourier(modulada_100,75)

plotter("figura2.5", "Señal 100 bits vs frecuencia Tbits = 1", "Frecuencia [Hz]", "Amplitud", ffreq_100, abs(fourier_100))

#Para tbits_100
t_10, t_100 = gen_time(tbits_1)
porta_75, porta_125 = gen_porta(t_10,t_100)
#Modulacion FSK
modulada_100 = modulador_FSK(array_100, porta_75, porta_125)

fourier_100,ffreq_100 = fourier(modulada_100,75)

plotter("figura2.5", "Señal 100 bits vs frecuencia Tbits = 1", "Frecuencia [Hz]", "Amplitud", ffreq_100, abs(fourier_100))


#Para tbits_100
t_10, t_100 = gen_time(tbits_100)
porta_75, porta_125 = gen_porta(t_10,t_100)
#Modulacion FSK
modulada_100 = modulador_FSK(array_100, porta_75, porta_125)

fourier_100,ffreq_100 = fourier(modulada_100,75)

plotter("figura3", "Señal 100 bits vs frecuencia Tbits = 1/50", "Frecuencia [Hz]", "Amplitud", ffreq_100, abs(fourier_100))

#Para tbits_1000
t_10, t_100 = gen_time(tbits_1000)
porta_75, porta_125 = gen_porta(t_10,t_100)
#Modulacion FSK
modulada_100 = modulador_FSK(array_100, porta_75, porta_125)

fourier_100,ffreq_100 = fourier(modulada_100,75)

plotter("figura4", "Señal 100 bits vs frecuencia Tbits = 1/100", "Frecuencia [Hz]", "Amplitud", ffreq_100, abs(fourier_100))

#Para tbits_10000
t_10, t_100 = gen_time(tbits_10000)
porta_75, porta_125 = gen_porta(t_10,t_100)
#Modulacion FSK
modulada_100 = modulador_FSK(array_100, porta_75, porta_125)

fourier_100,ffreq_100 = fourier(modulada_100,75)

plotter("figura5", "Señal 100 bits vs frecuencia Tbits = 1/200", "Frecuencia [Hz]", "Amplitud", ffreq_100, abs(fourier_100))

#Para tbits_20000
t_10, t_100 = gen_time(tbits_12000)
porta_75, porta_125 = gen_porta(t_10,t_100)
#Modulacion FSK
modulada_100 = modulador_FSK(array_100, porta_75, porta_125)

fourier_100,ffreq_100 = fourier(modulada_100,75)

plotter("figura6", "Señal 100 bits vs frecuencia Tbits = 1/250", "Frecuencia [Hz]", "Amplitud", ffreq_100, abs(fourier_100))

#Para tbits_20000
t_10, t_100 = gen_time(tbits_12500)
porta_75, porta_125 = gen_porta(t_10,t_100)
#Modulacion FSK
modulada_100 = modulador_FSK(array_100, porta_75, porta_125)

fourier_100,ffreq_100 = fourier(modulada_100,75)

plotter("figura6", "Señal 100 bits vs frecuencia Tbits = 1/300", "Frecuencia [Hz]", "Amplitud", ffreq_100, abs(fourier_100))


#Para tbits_20000
t_10, t_100 = gen_time(tbits_20000)
porta_75, porta_125 = gen_porta(t_10,t_100)
#Modulacion FSK
modulada_100 = modulador_FSK(array_100, porta_75, porta_125)

fourier_100,ffreq_100 = fourier(modulada_100,75)

plotter("figura6", "Señal 100 bits vs frecuencia Tbits = 1/500", "Frecuencia [Hz]", "Amplitud", ffreq_100, abs(fourier_100))

#Para tbits_20000
t_10, t_100 = gen_time(tbits_20300)
porta_75, porta_125 = gen_porta(t_10,t_100)
#Modulacion FSK
modulada_100 = modulador_FSK(array_100, porta_75, porta_125)

fourier_100,ffreq_100 = fourier(modulada_100,75)

plotter("figura6", "Señal 100 bits vs frecuencia Tbits = 1/500", "Frecuencia [Hz]", "Amplitud", ffreq_100, abs(fourier_100))

