# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 22:58:06 2022

@authors: Andrés Araya V. (19.961.739-7) & Felipe Cornejo I. (20.427.782-6)
"""

#Programa para el desarrollo de modulación AM y FM sobre señales de audio.

#######################
#####IMPORTACIONES#####
#######################
from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftshift
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy import signal
from scipy import integrate
#import sympy
import scipy as sp

######################
#####DEFINICIONES#####
######################


def fm_demod(x, df=1.0, fc=0.0):
    ''' Perform FM demodulation of complex carrier.

    Args:
        x (array):  FM modulated complex carrier.
        df (float): Normalized frequency deviation [Hz/V].
        fc (float): Normalized carrier frequency.

    Returns:
        Array of real modulating signal.
    '''

    # Remove carrier.
    n = sp.arange(len(x))
    rx = x*sp.exp(-1j*2*sp.pi*fc*n)

    # Extract phase of carrier.
    phi = sp.arctan2(sp.imag(rx), sp.real(rx))

    # Calculate frequency from phase.
    y = sp.diff(sp.unwrap(phi)/(2*sp.pi*df))

    return y

#Entrada: Nombre del archivo wav, con su extensión incluida
#Salidas: Arreglo de flotantes que representan la amplitud en cada punto, la frecuencia de la señal.
#Función que lee una señal wav y obtiene la amplitud y frecuencia de este.
def open_wav_file(nombre):
    freq,info = read(nombre)
    dimension = info[0].size
    
    if dimension == 1:
    	data = info
    else:
    	data = info[:,dimension-1]
        
    return data,freq

#Entrada: Nombre de la figura, Titulo a mostrar, label x, labely, vector x, vector y,  opcional : color de la linea a graficar
#Salida: muestra un grafico dependiendo de X y resultado Y
#Funcion que genera un grafico respecto a los parametros entregados
def plotter(figura, titulo, xlab, ylab, x, y,xlim=False,ylim=False,color="lightgreen"):
    plt.figure(figura, (8,4))
    #plt.figure(figura)
    plt.title(titulo)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if ylim != False:
        plt.ylim(ylim[0], ylim[1])
    if xlim != False:
        plt.xlim(xlim[0], xlim[1])
    plt.plot(x, y, color, linewidth = 0.7)
    plt.savefig(figura)
    plt.show()

#Entrada: Nombre de la figura, Titulo a mostrar, label x, labely, vector x, vector y
#Salida: muestra un Espectrograma dependiendo de X y resultado Y
#Funcion que genera un Espectrograma respecto a los parametros entregados
def espectro_plotter(figura, titulo, xlab, ylab, x, y):
    #plt.figure(nombre, tamaño, resolución)
    plt.figure(figura)
    plt.title(titulo)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.specgram(x,Fs=y)
    plt.show()
    
#Entrada:Arreglo de amplitudes de una señal, el periodo de la señal por su largo.
#Salida: Arreglo de amplitudes de la transformada de fourier, arreglo de frecuencias de la señal
#Funcion que calcula la transformada de fourier entregando el eje x e y
def fourier(data,time):
    fourier = np.fft.fft(data)
    aux = np.arange(-len(fourier)/2,len(fourier)/2)
    ffreq=fftshift(aux/time)
    return fourier,ffreq

#Entrada:Arreglo de amplitudes de una señal, la frecuencia de la señal.
#Salida: Arreglo de amplitudes de la transformada de fourier, arreglo de frecuencias de la señal
#Funcion que calcula la transformada de fourier entregando el eje x e y (se hizo esta copia ya que la primera no dejaba ingresar por parametro datos de tipo np.ndarray)
def fourier2(data,freq):
    largo = len(data)
    tiempo = largo/freq
    fourier = np.fft.fft(data)
    aux = np.arange(-len(fourier)/2,len(fourier)/2)
    #shift
    ffreq=fftshift(aux/tiempo)
    return fourier,ffreq

##########
##  AM  ##
##########
#Entrada: Arreglo de amplitudes de una señal, arreglo en el tiempo de las amplitudes, nuevo arreglo el cual se se interpolará la data, la frecuencia de la señal portadora seteada en 20k [Hz]
#Salida: Arreglo de amplitudes de la señal modulada con un indice de 1 y Arreglo de amplitudes de la señal modulada con un indice de 1.25
#Función moduladora en AM para una señal, la cual ocupará k = 1 y k = 1.25 devolviendo 2 modulaciones
def mod_AM(data,x,t,freq_porta = 20000):
    #Interpolación
    mt = np.interp(t, x, data)  

    ###############
    ## GRAFICO 2 ##
    ###############
    plotter('figura5','Audio_AM','Tiempo [s]','Amplitud [db]',t[:600],mt[:600],(0,0.0073),(-7000,7000),"red")
	
	#Obtención de wct utilizando el arreglo tiempo de la portadora.
    wct = 2 * np.pi * freq_porta * t

	#Obtención de la senal modulada.
    AM1 = mt * np.cos(wct)
    #Obtención de la senal modulada.
    AM1_5 = 1.25 * mt * np.cos(wct)
    
    return AM1,AM1_5

#Entrada: Arreglo de amplitudes de la señal modulada en AM, arreglo el cual indica el tiempo, frecuencia de la portadora, indice de modulación.
#Salida: Arreglo de amplitudes de la señal Demodulada
#Función demoduladora de AM
def demod_AM(signal, time, freq_porta, indice=1):
    signal = signal/indice
    portadora = np.cos(2 * np.pi * freq_porta * time)
    demodulada = signal * portadora
    return demodulada

#Entrada: Arreglo de amplitudes de la señal demodulada en AM, frecuencia de la señal
#Salida: Arreglo de amplitudes de la señal filtrada con pasabajos
#Función filtro pasabajos.
def filterr(data, freq):
    taps=1001
    nyq = freq/2
    corte=nyq*0.09
    coef_fir = signal.firwin(taps,corte/nyq, window = "hamming")
    filtrada = signal.lfilter(coef_fir,1.0,data)
    return filtrada

##########
##  FM  ##
##########

#Entrada: Arreglo de amplitudes de una señal, frecuencia de la señal moduladora, indice de modulación, la frecuencia de la señal portadora seteada en 20k [Hz]
#Salida: Arreglo de amplitudes de la señal modulada FM
#Función moduladora en FM para una señal
def mod_FM(data, freq, k, freq_porta = 20000):
    largo = len(data)
    tiempo = largo/freq
    x = np.linspace(0, tiempo, largo)
    f = freq/2 #frecuencia de muestreo, mitad de la frecuencia original || aqui está la parte de 2*fc
    
    t = np.linspace(0, int(tiempo), int(freq_porta*tiempo))
    mt = np.interp(t, x, data)# se define la funcion que representa al audio
    #definicion de la integral de la ecuacion
    integral = integrate.cumtrapz(mt, t, initial = 0)
    
    w = f * t
    #se modula en FM segun la ecuacion
    FM = np.cos(2 * w * np.pi + k * integral * np.pi)
    return FM
    
#############
###PARTE 1###
#############
#DEFINICION DE VARIABLES INICIALES
data, freq = open_wav_file("handel.wav")
largo = len(data)
time = largo/float(freq)
t = np.linspace(0,time,largo)
t2 = np.linspace(0,time,largo*10)
freq2 = freq * 10

freq_porta = 20000


###############
## GRAFICO 1 ##
###############
plotter("figura1", "Señal original", "Tiempo [s]", "Amplitud [db]", t, data)

#Interpola y devuelve una función con la relación t y data
interp = interp1d(t,data)
#Crea una muestra con la data a mostrar de largo -> largo*10
data_resample= interp(t2)



###############
## GRAFICO 2 ##
###############
fourier, ffreq = fourier(data_resample,time)
plotter("figura2", "T. de Fourier para Original", "Frecuencia [Hz]", "Amplitud [db]", ffreq, fourier, ylim=(0,2.5e8))

################ 
###PARTE 2 AM###
################

###############
## GRAFICO 3 ##
###############
#Se crea la señal portadora.
singal_porta = np.cos(2*np.pi*freq_porta*t2)
plotter('figura3','Señal Portadora AM','Tiempo [s]','Amplitud [db]',t2[:600],singal_porta[:600],(0,0.0073),(-1,1))

###############
## GRAFICO 4 ##
###############
fourier_porta, ffreq_porta = fourier2(singal_porta,freq2)
plotter("figura4", "T. de Fourier para Portadora", "Frecuencia [Hz]", "Amplitud [db]", ffreq_porta, fourier_porta, ylim=(0,2.8e5))

##################################
## MODULACION y DEMODULACIÓN AM ##
##################################
signal_AM1,signal_AM1_5 = mod_AM(data,t,t2)

###############
## GRAFICO 6 ##
###############
plotter('figura6','Señal Modulada AM con k = 1','Tiempo [s]','Amplitud [db]',t2[:600],signal_AM1[:600],(0,0.0073),(-7000,7000))

###############
## GRAFICO 7 ##
###############
plotter('figura7','Señal Modulada AM con k = 1.25','Tiempo [s]','Amplitud [db]',t2[:600],signal_AM1_5[:600],(0,0.0073),(-1000000000,1000000000))

###############
## GRAFICO 8 ##
###############
fourier_AM1, ffreq_AM1 = fourier2(signal_AM1,freq2)
plotter("figura8", "T. de Fourier para Señal AM con k = 1", "Frecuencia [Hz]", "Amplitud [db]", ffreq_AM1, fourier_AM1, ylim=(0,1e8))

###############
## GRAFICO 9 ##
###############
fourier_AM1_5, ffreq_AM1_5 = fourier2(signal_AM1_5,freq2)
plotter("figura9", "T. de Fourier para Señal AM con k = 1.25", "Frecuencia [Hz]", "Amplitud [db]", ffreq_AM1_5, fourier_AM1_5, ylim=(0,1.5e8))

################
## GRAFICO 10 ##
################
signal_AM1_dem = demod_AM(signal_AM1, t2, freq_porta,1)
plotter("figura10", "Señal Demodulada AM con k = 1", "Tiempo [s]", "Amplitud [db]", t2, signal_AM1_dem)

###############
## GRAFICO 11 ##
###############
fourier_AM1_dem, ffreq_AM1_dem = fourier2(signal_AM1_dem,freq2)
plotter("figura11", "T. de Fourier para Señal Demodulada AM, k = 1", "Frecuencia [Hz]", "Amplitud [db]", ffreq_AM1_dem, fourier_AM1_dem, ylim=(0,1.5e8))

################
## GRAFICO 12 ##
################
#Aplicación de Filtro pasabajos para obtener solo las frecuencias que se necesitan.
signal_AM1_dem = filterr(signal_AM1_dem,freq2)
plotter("figura12", "Señal Demodulada AM con k = 1", "Tiempo [s]", "Amplitud [db]", t2, signal_AM1_dem)

################
## GRAFICO 13 ##
################
fourier_AM1_dem, ffreq_AM1_dem = fourier2(signal_AM1_dem,freq2)
plotter("figura13", "T. de Fourier para Señal Demodulada AM F, k = 1", "Frecuencia [Hz]", "Amplitud [db]", ffreq_AM1_dem, fourier_AM1_dem, ylim=(0,1.5e8))

################
## GRAFICO 14 ##
################
signal_AM1_5_dem = demod_AM(signal_AM1_5, t2, freq_porta, 1.25)
plotter("figura14", "Señal Demodulada AM con k = 1.25", "Tiempo [s]", "Amplitud [db]", t2, signal_AM1_5_dem)

################
## GRAFICO 15 ##
################
fourier_AM1_5_dem, ffreq_AM1_5_dem = fourier2(signal_AM1_5_dem,freq2)
plotter("figura15", "T. de Fourier para Señal Demodulada AM, k = 1.25", "Frecuencia [Hz]", "Amplitud [db]", ffreq_AM1_5_dem, fourier_AM1_5_dem, ylim=(0,1.5e8))

################
## GRAFICO 16 ##
################
#Aplicación de Filtro pasabajos para obtener solo las frecuencias que se necesitan.
signal_AM1_5_dem = filterr(signal_AM1_5_dem,freq2)
plotter("figura16", "Señal Demodulada AM con k = 1.25", "Tiempo [s]", "Amplitud [db]", t2, signal_AM1_5_dem)

################
## GRAFICO 17 ##
################
fourier_AM1_5_dem, ffreq_AM1_5_dem = fourier2(signal_AM1_5_dem,freq2)
plotter("figura17", "T. de Fourier para Señal Demodulada AM F, k = 1.25", "Frecuencia [Hz]", "Amplitud [db]", ffreq_AM1_5_dem, fourier_AM1_5_dem, ylim=(0,1.6e8))


##################################
## MODULACION y DEMODULACIÓN FM ##
##################################

tiempo_resample = np.linspace(0, int(time), int(freq_porta*time))
port_resample = np.linspace(0, int(freq_porta), int(freq_porta*time))
senialPortadora = np.cos(2 * np.pi * port_resample)

senial_mod_FM1 = mod_FM(data, freq, k = 1)

#######################################################################################################
#######################################################################################################
'''
awita = fm_demod(senial_mod_FM1, df=1.0, fc=0.0)
taw= np.linspace(0, 10, 178497)
################
## GRAFICO XX ##
################
plotter('figuraXX', 'DEMOD FM', "Tiempo [s]", 'Amplitud [db]', taw, awita, ylim=(-1,1))

###############
## GRAFICO 1 ##
###############
plotter("figura1", "Señal ORIGINAL XX", "Tiempo [s]", "Amplitud [db]", t, data)
#######################################################################################################
#######################################################################################################
'''
################
## GRAFICO 18 ##
################
plotter('figura18', 'Señal Modulada FM con k = 1', 'Tiempo [s]', 'Amplitud [db]', tiempo_resample[:600], senial_mod_FM1[:600], (0,0.027),(-1, 1))

################
## GRAFICO 19 ##
################
fourier_FM1, ffreq_FM1 = fourier2(senial_mod_FM1,freq2)
plotter("figura19", "T. de Fourier para Señal FM con k = 1", "Frecuencia [Hz]", "Amplitud [db]", ffreq_FM1, fourier_FM1, ylim=(0,1500))

senial_mod_FM2 = mod_FM(data, freq, k = 1.25)

################
## GRAFICO 20 ##
################
plotter('figura20', 'Señal Modulada FM con k = 1.25', 'Tiempo [s]', 'Amplitud [db]', tiempo_resample[:600], senial_mod_FM2[:600], (0,0.027),(-1, 1))

################
## GRAFICO 21 ##
################
fourier_FM2, ffreq_FM2 = fourier2(senial_mod_FM2,freq2)
plotter("figura21", "T. de Fourier para Señal FM con k = 1.25", "Frecuencia [Hz]", "Amplitud [db]", ffreq_FM2, fourier_FM2, ylim=(0,1500))

#definicion de la integral de la ecuacion
mt = np.interp(t2, t, data)# se define la funcion que representa al audio
integral = integrate.cumtrapz(mt, t2, initial = 0)
t = np.linspace(0, int(time), int(freq_porta*time))
w = freq/2 * t2
#se modula en FM segun la ecuacion
FM_1 = np.cos(w * np.pi + 1 * integral * np.pi)

#DERIVACION DE LA SEÑAL MODULADA EN FM | NO FUNCIONÓ EL GRÁFICO, PARA PROBAR SOLO DESCOMENTE ESTA SECCIÓN
'''
function_FM = IUS(t2,FM_1)
dfunction_FM = function_FM.derivative()
print(dfunction_FM)
plotter("figura22", "Derivada de la señal en FM", "Frecuencia [Hz]", "Amplitud [db]", t2, function_FM)
'''
####################
## ANCHO DE BANDA ##
####################

########
## AM ##
########
#ANCHO DE BANDA AM
bw_AM = (freq_porta+freq) - (freq_porta-freq)
print("El Ancho de banda de la señal modulada AM es: ", bw_AM, "\n\n")

########
## FM ##
########
#ANCHO DE BANDA FM
#Para k = 1
k_1 = 1
deltaF_1 = k_1 * freq
bw_FM1 = 2 * (deltaF_1 + freq)
print("El Ancho de banda de la señal modulada FM y k = 1, es: ", bw_FM1, "\n\n")

k_1_5 = 1.25
deltaF_1_5 = k_1_5 * freq
bw_FM1_5 = 2 * (deltaF_1_5 + freq)
print("El Ancho de banda de la señal modulada FM y k = 1.25, es: ", bw_FM1_5, "\n\n")

