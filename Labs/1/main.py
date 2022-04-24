from scipy.signal import butter, lfilter
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft,fftfreq,ifft

#Entrada: Nombre de la figura, Titulo a mostrar, label x, labely, vector x, vector y,  opcional : color de la linea a graficar
#Salida: muestra un grafico dependiendo de X y resultado Y
#Funcion que genera un grafico respecto a los parametros entregados
def plotter(figura, titulo, xlab, ylab, x, y, color="gold"):
    #plt.figure(nombre, tamaño, resolución)
    plt.figure(figura)
    plt.title(titulo)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.plot(x, y, color, linewidth=0.5)
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
    
#############
## PARTE 2 ##
#############

# *_f -> frecuencia de los audios en formato wav
# *_audio -> arreglo de amplitudes de handel.wav y tipo de dato
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html
cr_f, cr_audio = wavfile.read("AudioCR.wav")
fc_f, fc_audio = wavfile.read("AudioFC1 (1).wav")
print("Frecuencia de Carlos Retamales: ", cr_f, "[Hz]")
print("Frecuencia de Felipe Cornejo: ", fc_f, "[Hz]")

# Como *_audio contiene numero de amplitudes
cr_amps = len(cr_audio)
fc_amps = len(fc_audio)

print("Amplitudes de Carlos Retamales: ", cr_amps)
print("Amplitudes de Felipe Cornejo: ", fc_amps)

# Arreglo con valores de tiempo para cada amplitud [Coordenada X en los siguientes Gráficos]
cr_Time = np.linspace(0, cr_amps/cr_f, cr_amps)
fc_Time = np.linspace(0, fc_amps/fc_f, fc_amps)

#############
## PARTE 3 ##
#############

###############
## GRAFICO 1 ##
###############
plotter("Grafico 1 - Amplitud Carlos Retamales", 'Audio original Carlos Retamales en el tiempo', "Tiempo [s]", "Amplitud [db]", cr_Time, cr_audio)
plotter("Grafico 1 - Amplitud Felipe Cornejo", 'Audio original Felipe Cornejo en el tiempo', "Tiempo [s]", "Amplitud [db]", fc_Time, fc_audio,"lightgreen")

#############
## PARTE 4 ##
#############

# Obtener las transformadas de Fourier usando fft de Scipy
# https://docs.scipy.org/doc/scipy/reference/tutorial/fft.html
cr_Fourier = fft(cr_audio)
fc_Fourier = fft(fc_audio)

# Obtener las frecuencias por cada punto de la transformada de fourier
# https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.fftfreq.html
cr_Fourier_freq = fftfreq(cr_amps)
fc_Fourier_freq = fftfreq(fc_amps)

# Obtener las transformada de Fourier inversa (solo la parte real)
cr_Fourier_inv = ifft(cr_Fourier).real
fc_Fourier_inv = ifft(fc_Fourier).real

print(fc_Fourier_inv == fc_audio)

###############
## GRAFICO 2 ##
###############
plotter("Grafico 2 - Transformada Carlos Retamales", "Transfromada de Fourier para Carlos Retamales", "Frecuencia [Hz]", "Amplitud [dB]",cr_Fourier_freq , abs(cr_Fourier))
plotter("Grafico 2 - Transformada Felipe Cornejo", "Transfromada de Fourier para Felipe Cornejo", "Frecuencia [Hz]", "Amplitud [dB]",  fc_Fourier_freq, abs(fc_Fourier),"lightgreen")

###############
## GRAFICO 3 ##
###############
plotter("Grafico 3 - Transformada Inversa Carlos Retamales", "Transfromada de Fourier Inversa para Carlos Retamales", "Tiempo [s]", "Amplitud [dB]", cr_Time, cr_Fourier_inv)
plotter("Grafico 3 - Transformada Inversa Felipe Cornejo", "Transfromada de Fourier Inversa para Felipe Cornejo", "Tiempo [s]", "Amplitud [dB]", fc_Time, fc_Fourier_inv, "lightgreen")

# Está bien que quede igual a los graficos 1, ya que se aplicó la inversa a la transformada de fourier de ambas señales

#############
## PARTE 5 ##
#############

# Obtener el espectrograma para cada uno de los auidos
# Desde la función espectro_plotter, plt.especgram(x,y) en y existe una variable opcional que puede sacar las frecuencias
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.specgram.html

###############
## GRAFICO 4 ##
###############
espectro_plotter("Grafico 4 - Espectograma de Carlos Retamales", "Espectograma del audio de Carlos Retamales", 'Tiempo [s]', 'Frecuencia [Hz]', cr_audio, cr_f)
espectro_plotter("Grafico 4 - Espectograma de Felipe Cornejo", "Espectograma del audio de Felipe Cornejo", 'Tiempo [s]', 'Frecuencia [Hz]', fc_audio, fc_f)

#############
## PARTE 7 ##
#############

###############
## GRAFICO 5 ##
###############
# Carga de ruido cafe para modificación de auido de Felipe Cornejo
noise_f, noise_audio = wavfile.read("brown_noise.wav")
print("frecuencia del ruido: ", noise_f)
noise_audio = noise_audio /2
noise_audio = noise_audio[:,0]
noise_amp = len(noise_audio)
noise_Time = np.linspace(0, noise_amp/noise_f, noise_amp)
plotter("Grafico 5 - Amplitud Ruido cafe", 'Amplitud Ruido cafe', "Tiempo [s]", "Amplitud [db]", noise_Time, noise_audio)


#Se añaden las amplitudes al audio de felipe
fc_noise = np.zeros(fc_amps)
i = 0
while i<len(noise_audio):
    fc_noise[i] = fc_audio[i] + noise_audio[i]
    i+=1
while i<len(fc_noise):
    fc_noise[i] = fc_audio[i]
    i+=1

fc_noise_amps = len(fc_noise)
fc_noise_Time = np.linspace(0, fc_noise_amps/fc_f, fc_noise_amps)
    
wavfile.write("Felipe_brownnoise - 1.wav", fc_f, fc_noise.astype(np.int16))

###############
## GRAFICO 6 ##
###############

plotter("Grafico 6 - Amplitud Felipe Cornejo + brown noise", 'Audio con Ruido cafe Felipe Cornejo en el tiempo', "Tiempo [s]", "Amplitud [db]", fc_noise_Time, fc_noise,"lightgreen")

# Obtener las transformadas de Fourier usando fft de Scipy
# https://docs.scipy.org/doc/scipy/reference/tutorial/fft.html
fc_noise_Fourier = fft(fc_noise)

# Obtener las frecuencias por cada punto de la transformada de fourier
# https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.fftfreq.html
fc_noise_Fourier_freq = fftfreq(fc_noise_amps)

# Obtener las transformada de Fourier inversa (solo la parte real)
fc_noise_Fourier_inv = ifft(fc_noise_Fourier).real

###############
## GRAFICO 7 ##
###############

plotter("Grafico 7 - Transformada Felipe Cornejo + brown noise", "Transfromada de Fourier para Felipe Cornejo + brown noise", "Frecuencia [Hz]", "Amplitud [dB]",fc_noise_Fourier_freq , abs(fc_noise_Fourier), color="lightgreen")

###############
## GRAFICO 8 ##
###############

plotter("Grafico 8 - Transformada Inversa Felipe Cornejo + brown noise", "Transfromada Inversa de Fourier para Felipe Cornejo + brown noise", "Frecuencia [Hz]", "Amplitud [dB]", fc_noise_Time, fc_noise_Fourier_inv, color="lightgreen")

###############
## GRAFICO 9 ##
###############

espectro_plotter("Grafico 9 - Espectograma de brown noise", "Espectograma del audio de brown noise", 'Tiempo [s]', 'Frecuencia [Hz]', noise_Time, noise_f)

################
## GRAFICO 10 ##
################

espectro_plotter("Grafico 10 - Espectograma de Felipe Cornejo + brown noise", "Espectograma del audio de Felipe Cornejo + brown noise", 'Tiempo [s]', 'Frecuencia [Hz]', fc_noise, fc_f)

#############
## PARTE 8 ##
#############

##############
## FILTRO 1 ##
##############

# Para observar la efectividad del filtro se aplicará para solo un rango de tiempo.

inicio = 2000
fin = 18000

b1, a1 = butter(N=8, Wn=1 * inicio/fc_f, btype='highpass')
b2, a2 = butter(N=8, Wn=4 * inicio/fc_f, btype='highpass')
b3, a3 = butter(N=8, Wn=8 * inicio/fc_f, btype='highpass')

afiltrado1 = lfilter(b1, a1, fc_noise)
afiltrado2 = lfilter(b2, a2, fc_noise)
afiltrado3 = lfilter(b3, a3, fc_noise)

################
## GRAFICO 11 ##
################

espectro_plotter("Grafico 11 - Espectograma de Felipe Cornejo + brown noise con filtro1", "Espectograma del audio de Felipe Cornejo + brown noise con filtro1", 'Tiempo [s]', 'Frecuencia [Hz]', afiltrado1, fc_f)

################
## GRAFICO 12 ##
################

espectro_plotter("Grafico 12 - Espectograma de Felipe Cornejo + brown noise con filtro2", "Espectograma del audio de Felipe Cornejo + brown noise con filtro2", 'Tiempo [s]', 'Frecuencia [Hz]', afiltrado2, fc_f)

################
## GRAFICO 13 ##
################

espectro_plotter("Grafico 13 - Espectograma de Felipe Cornejo + brown noise con filtro3", "Espectograma del audio de Felipe Cornejo + brown noise con filtro3", 'Tiempo [s]', 'Frecuencia [Hz]', afiltrado3, fc_f)

wavfile.write("AudioFiltrado - 1.wav", fc_f, afiltrado1.astype(np.int16))
wavfile.write("AudioFiltrado - 2.wav", fc_f, afiltrado2.astype(np.int16))
wavfile.write("AudioFiltrado - 3.wav", fc_f, afiltrado3.astype(np.int16))

b1, a1 = butter(N=8, Wn= fin/fc_f, btype='lowpass')
afiltrado4 = lfilter(b1, a1, afiltrado3)
wavfile.write("AudioFiltrado - 4.wav", fc_f, afiltrado4.astype(np.int16))
espectro_plotter("Grafico 14 - Espectograma de Felipe Cornejo + brown noise con filtro4", "Espectograma del audio de Felipe Cornejo + brown noise con filtro4", 'Tiempo [s]', 'Frecuencia [Hz]', afiltrado4, fc_f)



