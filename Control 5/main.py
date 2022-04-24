# -*- coding: utf-8 -*-
'''
LABORATORIO 4: MODULACIÓN DIGITAL
Aylin Rodrí­guez
Jennifer Velozo
'''

######################## LIBRERIAS ########################
import matplotlib.pyplot as plt
import numpy as np

######################## FUNCIONES ########################
#Funciónn para graficar los bits en el dominio del tiempo
#ENTRADA: el arreglo de bits
#SALIDA: gráfico
def graficarBits(bits, flag):
    #Se grafican los bits en el tiempo
    lim = len(bits)/100 
    dim = 100
    vectorBits = []
    for i in range (0,len(bits)):
        f = np.ones(dim)
        x = f * bits[i]
        vectorBits = np.concatenate((vectorBits,x))
    
    dim2 = len(vectorBits)
    t = np.linspace(0,lim,dim2)
    plt.figure()
    if flag == 1: 
        plt.title("Señal original")
    else:
        plt.title("Señal demodulada")
    plt.xlabel("Tiempo [s]")
    plt.plot(t, vectorBits)
    plt.grid(True)
    plt.show()
    

#Función que realiza la modulación FSK
#ENTRADAS: la señal de bits y la tasa de bit
#SALIDA: tiempo de la señal modulada, la señal modulada, y el largo de un bit
def FSK(bits, bitrate):   
    f1 = bitrate
    f2 = 2 * f1
    periodo = 1 / f1
    print("Frecuencia para bit 0: ", f1)
    print("Frecuencia para bit 1: ", f2)
    
    largobit = int(periodo * 10 * f2) 
    tiempo = np.linspace(0, periodo, num = largobit)
    #print("Num muestras en 1 bit: ", largobit)
    
    senal = []
    cero = np.cos(2 * np.pi * f1 * tiempo)
    uno = np.cos(2 * np.pi * f2 * tiempo)
    for bit in bits:
        if bit == 0:
            senal.extend(cero)
        else:
            senal.extend(uno)

    tiempoT = len(bits) * periodo  
    tiemposenal = np.linspace(0, tiempoT, len(senal))
    
    #print("Tiempo total de la senal: ", tiempoT)
    #print("Numero de muestras de la senal: ", len(senal), "\n")
    return tiemposenal, senal, largobit



#Función encargada de realizar demodulaciónn FSK
#ENTRADA: tiempo de bit, la señal, el nro de muestras de un bit, la tasa de bit
#SALIDA: señal demodulada, es decir, el arreglo de bits
def DFSK(tiempo, senal, largobit, bitrate):
    f1 = bitrate
    f2 = 2 * f1
    cosf1 = np.cos(2 * np.pi * f2 * tiempo)
    #plt.plot(tiempo, senal)
    dembit1 = senal * cosf1
    #print("Media de un bit: ", np.mean(dembit1))
    #print(senal)
    #print(dembit1)
    n_bits = int(len(dembit1) / largobit)
    #print("N_bits:",n_bits)
    #plt.plot(tiempo, dembit1)
    
    
    bitsdemod = [] 
    #print("largo bit:", largobit)
    for i in range(1, n_bits + 1):
        voltaje = dembit1[((i - 1) * largobit): i * largobit - 1]
        
        #print("Voltaje:",voltaje)
        mediaV = np.mean(voltaje)
        #print("Media:", mediaV)
        if mediaV > 0.25:
            bitsdemod.append(1)
        else:
            bitsdemod.append(0)
    return bitsdemod


#Función que agrega el ruido AWGN
#ENTRADAS: la señal, y el nro de muestras que tienen un bit
#SALIDA: la señal con ruido agregado
def ruido(senal, snr):
    ruido = np.random.normal(0, 1, len(senal))
    energia_s = np.sum(np.abs(senal) * np.abs(senal))
    energia_n = np.sum(np.abs(ruido) * np.abs(ruido))
    snr_lineal = np.exp(snr/10)
    sigma = np.sqrt(energia_s / (energia_n * snr_lineal))
    print('Desviación ruido: ' + str(sigma))
    ruido = sigma * ruido
    awgn = senal + ruido
    return awgn

#Función que calcula la tasa de error binario en la señal demodulada.
#Entrada: señal demodulada y señal
#Salida: tasa de error
def error(bitsdemod, bits):
    contador = 0
    for i in range(len(bits)):
        if bits[i] != bitsdemod[i]:
            contador += 1
    print("Numero de errores en la transmisión: ",contador)
    ber = float(contador / len(bits))
    print("Tasa de error binario: ",ber)
    return ber

#Función encargarda de simular el canal con ruido
#ENTRADA: tasa de bit
def simulacionCanal( bitrate ):
    largosenal = 1e5 # Se puede cambiar a conveniencia
    bits = np.random.randint(2, size = int(largosenal))
    colores = ['-b', '-g', '-r']
    
    plt.figure(1)
    for i in range(0, 3):
        snr_x = []
        ber_y = []
        bitrate = bitrate + i*1000 
        tiempo, senal, len_bit = FSK(bits, bitrate)
        for snr in range(-2, 11, 1):
            print("##### Prueba SNR = {}[dB] para bitrate = {}[bits/s] #####".format(snr, bitrate))
            awgn = ruido(senal, snr)
            demod = DFSK(tiempo, awgn, len_bit, bitrate) 
            ber = error(demod, bits)
            snr_x.append(snr)
            ber_y.append(ber)
            lab = str(bitrate) + ' [bps]'
            print("##### Fin prueba #####\n")
            
        plt.plot(snr_x, ber_y, colores[i], label=lab, marker="o")

    plt.grid(True)
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.yscale('log')
    plt.xscale('linear')
    plt.title('Rendimiento del sistema de comunicación')
    plt.legend()
    plt.show()


######################## BLOQUE PRINCIPAL ########################
print('----- INICIANDO BLOQUE DE PRUEBA -----')
bits = [0,0,0,1,1,1,1,0,1,1,0,0,0,0,1,1]
graficarBits(bits,1)
print('Bits de prueba:', bits)

bitrate = 100
print('Bitrate de prueba:', bitrate)

print('\n-----MODULACIÓNN FSK-----')
time, senal, largobit = FSK(bits, bitrate)
plt.title("Señal modulada en FSK")
plt.xlabel("Tiempo [s]")
plt.plot(time,senal)
plt.grid()
plt.show()
f1 = bitrate
f2 = 2 * f1
cosf1 = np.cos(2 * np.pi * f2 * time)
#plt.plot(tiempo, senal)
dembit1 = senal * cosf1
plt.title("Voltaje de la señal modulada")
plt.plot(time, dembit1)
plt.xlabel("Tiempo [s]")
plt.ylabel("Voltaje [volts]")
plt.show()

senalRuido = ruido(senal, 3)
plt.title("Señal modulada en FSK con ruido SNR = 3")
plt.xlabel("Tiempo [s]")
plt.plot(time,senalRuido)
plt.grid()
plt.show()
senalRuido = ruido(senal, 30)
plt.title("Señal modulada en FSK con ruido SNR = 30")
plt.xlabel("Tiempo [s]")
plt.plot(time,senalRuido)
plt.grid()
plt.show()



print('\n----- DEMODULACIÓN FSK -----')
demod = DFSK(time, senal, largobit, bitrate)
print('Bits de Salida Demodulados:', demod)
graficarBits(demod, 2)
errorBits = error(demod, bits)

print('----- FIN DE BLOQUE DE PRUEBA -----\n')

print('----- SIMULACION DE CANAL AWGN -----')
bitrate = 1000 #Se puede cambiar a conveniencia 
simulacionCanal(bitrate)

