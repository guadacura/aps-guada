# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 15:43:45 2025

@author: USUARIO
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal #para hacer la señal cuadrada

fm=100000 #frecuencia de muestreo
N=500    #cantidad de muestras
fs=2000   #frecuencia de la señal
dt=1/fm  #tiempo entre muestras

#--------------------------------------------------
#Seno
def funcion_seno(vc=1, dc=0, fs=None, ph=0, nn=N, fm=fm):
    t = np.linspace(0, N/fm, N, endpoint=False)
    sen=(vc*np.sin(2*np.pi*fs*t+ph)+dc)
    return t,sen
tt, yy = funcion_seno(vc=1, dc=0, fs=fs, ph=0, nn=N, fm=fm)

energiasen = np.sum(yy**2)*dt
muestras=len(yy)
tiempoMuestras= dt
print(f'La energía de la función senoidal es: {energiasen:.4f}V^2⋅s.')
print(f'La cantidad de muestras es: {muestras}')
print(f'El tiempo entre muestras es: {tiempoMuestras}s')


#--------------------------------------------------
#desfasada
tt1, yy1 = funcion_seno(vc=3, dc=0, fs=fs, ph=np.pi/2, nn=N, fm=fm)

energiades = np.sum(yy1**2)*dt
muestras=len(yy1)
tiempoMuestras= dt
print(f'La energía de la función senoidal es: {energiades:.4f}V^2⋅s.')
print(f'La cantidad de muestras es: {muestras}')
print(f'El tiempo entre muestras es: {tiempoMuestras}s')

#--------------------------------------------------
#Modulada
def funcion_seno_moduladora(vc=1, dc=0, fs2=None, ph=0, nn=N, fm=fm):
    t = np.linspace(0, N/fm, N, endpoint=False)
    sen=(vc*np.sin(2*np.pi*fs2*t+ph)+dc)
    return t,sen

fs2=1000
tt2, yy2 = funcion_seno_moduladora(vc=1, dc=0, fs2=1000, ph=0, nn=N, fm=fm)
mody=yy2*yy

energiamod = np.sum(mody**2)*dt
muestras=len(mody)
tiempoMuestras= dt
print(f'La energía de la función modulada es: {energiamod:.4f}V^2⋅s.')
print(f'La cantidad de muestras es: {muestras}')
print(f'El tiempo entre muestras es: {tiempoMuestras}s')
#--------------------------------------------------
#recortada
vc=1
potencia=(vc**2)/2

#Calulo el 75% de la potencia
threshold=potencia*0.75

yyRecortada=np.clip(mody,-threshold,threshold)

energiarec = np.sum(yyRecortada**2)*dt
muestras=len(yyRecortada)
tiempoMuestras= dt
print(f'La energía de la función modulada recortada: {energiarec:.4f}V^2⋅s.')
print(f'La cantidad de muestras es: {muestras}')
print(f'El tiempo entre muestras es: {tiempoMuestras}s')
#--------------------------------------------------
#Cuadrada
t = np.linspace(0, N/fm, N, endpoint=False)
frec=4000

# Señal cuadrada
sq_wave = signal.square(2 * np.pi * frec * t)

energiacuad = np.sum(sq_wave**2)*dt
muestras=len(sq_wave)
tiempoMuestras= dt
print(f'La energía de señal cuadrada: {energiacuad:.4f}V^2⋅s.')
print(f'La cantidad de muestras es: {muestras}')
print(f'El tiempo entre muestras es: {tiempoMuestras}s')

#--------------------------------------------------
#Pulsoo
# Parámetros
fm = 100000        # frecuencia de muestreo [Hz]
A = 1.0           # amplitud del pulso
t0 = 0       # inicio del pulso [s]
Tp = 10e-3        # duración del pulso [s]
dt2=1/fm

# Vector de tiempo
t = np.linspace(0, N/fm, N, endpoint=False)

# Defino el pulso
pulso = np.zeros_like(t)                   # todo en cero
pulso[(t >= t0) & (t < t0 + Tp)] = A       # intervalo activo

energiapul = np.sum(pulso**2)*dt2
muestras=len(pulso)
tiempoMuestras= dt2
print(f'La energía del pulso es: {energiapul:.4f}V^2⋅s.')
print(f'La cantidad de muestras es: {muestras}')
print(f'El tiempo entre muestras es: {tiempoMuestras}s')

#----------------------------------------------------
#grafico
def funcion_grafico(x, y, titulo, ejex, ejey):  
    plt.plot(x, y, label=f'{fs} Hz')
    plt.title(titulo)
    plt.xlabel(ejex)
    plt.ylabel(ejey)
    plt.legend(title="Frecuencia")
    plt.grid(True)
    plt.show()

#---------------------------------------------------


def funcion_LTI(x):
    N = len(x)          # longitud de la señal de entrada
    y = np.zeros(N)     # inicializo la salida en ceros

    for n in range(N):
        if n >= 2:
            y[n] = 0.03*x[n] + 0.05*x[n-1] + 0.03*x[n-2] + 1.5*y[n-1] - 0.5*y[n-2]
        elif n == 1:
            y[n] = 0.03*x[n] + 0.05*x[n-1] + 1.5*y[n-1]
        else:  # n == 0
            y[n] = 0.03*x[n]
    
    return y



senoidalLti= funcion_LTI(yy)
longitud=len(senoidalLti)
print(f'La longitud de y es {longitud}')

desfasadaLti=funcion_LTI(yy1)
moduladaLti=funcion_LTI(mody)
recortadaLti=funcion_LTI(yyRecortada)
cuadradaLti=funcion_LTI(sq_wave)
pulsoLti=funcion_LTI(pulso)

funcion_grafico(tt, senoidalLti, 'Señal de salida de la senoidal', 'Tiempo[s]', 'Amplitud[V]')
funcion_grafico(tt, desfasadaLti, 'Señal de salida de la senoidal desfasada', 'Tiempo[s]', 'Amplitud[V]')
funcion_grafico(tt, moduladaLti, 'Señal de salida de la modulada', 'Tiempo[s]', 'Amplitud[V]')
funcion_grafico(tt, recortadaLti, 'Señal de salida de la recortada', 'Tiempo[s]', 'Amplitud[V]')
funcion_grafico(tt, cuadradaLti, 'Señal de salida de la cuadrada', 'Tiempo[s]', 'Amplitud[V]')
funcion_grafico(tt, pulsoLti, 'Señal de salida del pulso', 'Tiempo[s]', 'Amplitud[V]')

#------------------------------------
#pulso

pulso2= np.zeros(N)
pulso2[0]=1.0

pulso2Lti=funcion_LTI(pulso2)
funcion_grafico(tt, pulso2Lti, 'Señal de salida del segundo pulso', 'Tiempo[s]', 'Amplitud[V]')

#-----------------------------------
#Convolución con cada señal para obtener la misma salida

senConv=np.convolve(yy,pulso2Lti)[:N]
desConv=np.convolve(yy1,pulso2Lti)[:N]
modConv=np.convolve(mody,pulso2Lti)[:N]
recConv=np.convolve(yyRecortada,pulso2Lti)[:N]
sqrConv=np.convolve(sq_wave,pulso2Lti)[:N]
pulsConv=np.convolve(pulso,pulso2Lti)[:N]

funcion_grafico(tt, senConv, 'Señal de salida de la senoidal por convolución', 'Tiempo[s]', 'Amplitud[V]')
funcion_grafico(tt, desConv, 'Señal de salida de la senoidal desfasada por convolución', 'Tiempo[s]', 'Amplitud[V]')
funcion_grafico(tt, modConv, 'Señal de salida de la modulada por convolución', 'Tiempo[s]', 'Amplitud[V]')
funcion_grafico(tt, recConv, 'Señal de salida de la recortada por convolución', 'Tiempo[s]', 'Amplitud[V]')
funcion_grafico(tt, sqrConv, 'Señal de salida de la cuadrada por convolución', 'Tiempo[s]', 'Amplitud[V]')
funcion_grafico(tt, pulsConv, 'Señal de salida del pulso por convolución', 'Tiempo[s]', 'Amplitud[V]')


#----------------------
def funcion_LTI1(x):
    N = len(x)          # longitud de la señal de entrada
    y1 = np.zeros(N)     # inicializo la salida en ceros

    for n in range(N):
        if n >= 10:
            y1[n] = x[n] +3*x[n-10] 
        else:  # n == 0
            y1[n] = x[n]
    
    return y1

def funcion_LTI2(x):
    N = len(x)          # longitud de la señal de entrada
    y2 = np.zeros(N)     # inicializo la salida en ceros

    for n in range(N):
        y2[n] = x[n] +3*y2[n-10] 

    return y2

pulso3Lti=funcion_LTI1(pulso2)
pulso4Lti=funcion_LTI2(pulso2)
funcion_grafico(tt, pulso3Lti, 'Señal de salida del pulso en y[n]=x[n]+3⋅x[n−10] ', 'Tiempo[s]', 'Amplitud[V]')
funcion_grafico(tt, pulso4Lti, 'Señal de salida del pulso en y[n]=x[n]+3⋅y[n−10]', 'Tiempo[s]', 'Amplitud[V]')

#----------------------
#convoluciono por senoidal
senConv=np.convolve(yy,pulso3Lti)[:N]
desConv=np.convolve(yy,pulso4Lti)[:N]

funcion_grafico(tt, senConv, 'Señal de  salida de y[n]=x[n]+3⋅x[n−10] con seoindal por convolución', 'Tiempo[s]', 'Amplitud[V]')
funcion_grafico(tt, desConv, 'Señal de salida de y[n]=x[n]+3⋅y[n−10] con senoidal por convolución', 'Tiempo[s]', 'Amplitud[V]')

#--------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# Parámetros fisiológicos aproximados
R = 0.02       # mmHg·s/mL
C = 1.8        # mL/mmHg
Ts = 0.001     # tiempo de muestreo (s)
N = 2000       # cantidad de muestras (2 s de simulación)

# Eje temporal
t = np.arange(N) * Ts

# Flujo de entrada Q(t): pulso senoidal positivo (60 lat/min)
Q = 80 * (np.sin(2*np.pi*1*t) > 0) * np.sin(2*np.pi*1*t)

# Vector de presión
P = np.zeros(N)

# Ecuación en diferencias (Euler hacia adelante)
for n in range(N-1):
    P[n+1] = (1 - Ts/(R*C)) * P[n] + (Ts/C) * Q[n]

# Gráfica
plt.figure(figsize=(10,5))
plt.plot(t, Q, label="Flujo Q(t)", linestyle="--")
plt.plot(t, P, label="Presión P(t)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.title("Modelo Windkessel discretizado con parámetros fisiológicos")
plt.legend()
plt.grid(True)
plt.show()

