import numpy as np
from math import e
import matplotlib.pyplot as plt
from scipy.integrate import quad

def plotter(figura, titulo, xlab, ylab, x, y, color="gold"):
    #plt.figure(nombre, tamaño, resolución)
    plt.figure(figura)
    plt.title(titulo)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.plot(x, y, color,linewidth=1)
    plt.show()

A = 82
w = np.linspace(-1000,1001,1000)
fourier_part1 = (A*(e**(-1j*w*2)))/(w**2)
fourier_part2 = -A/(2*w**2)
fourier_part3 = (A*1j*(e**(-1j*w*3)))/(2*w)
fourier_part4 = -(A*(e**(-1j*w*3)))/(2*w**2)
fourier_total = fourier_part1 + fourier_part2 + fourier_part3 + fourier_part4

plotter("Transformada de Fourier para f(t)", "Transformada de Fourier para f(t)", "frecuencia", "F(w)", w, fourier_total.real)


#Calcular la energía de la señal en el dominio del tiempo y de la frecuencia
#Se debe dividir, asi como en el la parte (a) la función en 2 partes
#f1 = (A/2)*t , t e [0,2]
#f2 = (-A/2)*t + 2*A , t e [2,3]
# luego la energia respecto al tiempo será la integral del valor absoluto de la función al cuadrado.

arg1 = lambda t: abs((A/2)*t)**2
energia1 = quad(arg1, 0,2)[0]

arg2 = lambda t: abs((-A/2)*t + 2*A)**2
energia2 = quad(arg2,2,3)[0]
# Se saca el primer elemento del retorno de quad, ya que devuelve una array con primer elemento el resultado.

energia_total = energia1 + energia2
print("La energia total de la señal, respecto al tiempo es: ", energia_total)

#Por otro lado para la energia respecto a la frecuencia, se trabaja con la transformada anteriormente desarrollada
fourierw = lambda w: abs((A*(e**(-1j*w*2)))/(w**2) - A/(2*w**2) + (A*1j*(e**(-1j*w*3)))/(2*w) - (A*(e**(-1j*w*3)))/(2*w**2))**2
energia_freq = quad(fourierw, -1000,1001, limit=1000)[0]

print("El resultado de la Energía respecto a la frecuencia es: ", energia_freq)

#Para que se cumpla la ley de Parseval se debe dividir la energia por 2*pi
print("La energia finalmente es: ", energia_freq/(2*np.pi))

#No da exacto lo mismo pero se acercan por decimales al mismo numero, las razones de por que no da exacto puede ser 
#por la cantidad de divisiones que el programa tiene que hacer, junto con las integrales el cual su calculo retorna como
#segundo valor un factor de error tambien.

