# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 20:11:20 2025

@author: USUARIO
"""

import numpy as np
import matplotlib.pyplot as plt

fm=100000 #frecuencia de muestreo
N=500     #cantidad de muestras
fs=2000   #frecuencia de la señal
tiempoMuestras=1/fm  #tiempo entre muestras



tt=np.arange(N)/ fm
yy=(np.sin(2*np.pi*fs*tt))
 


dt = 1/fm
energia = np.sum(yy**2)*dt
muestras=len(yy)
tiempoMuestras= dt
print(f'La energía de la función senoidal desfasada es: {energia:.4f}')
print(f'La cantidad de muestras es: {muestras}')
print(f'El tiempo entre muestras es: {tiempoMuestras}s')

plt.figure(1)
plt.plot(tt, yy, label='{f} Hz')
plt.title('Señal Senoidal con Frecuencia 2KHZ')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.grid(True)
plt.show()

#----------------------------------------------------
#LA FUNCION DEFASADA Y AMPLIFICADA EN PI/2
yy1 = (np.sin(2*np.pi*fs*tt+np.pi/2))

dt = 1/fm
energia = np.sum(yy1**2)*dt
muestras=len(yy1)
tiempoMuestras= dt
print(f'La energía de la función senoidal desfasada es: {energia:.4f}')
print(f'La cantidad de muestras es: {muestras}')
print(f'El tiempo entre muestras es: {tiempoMuestras}s')

plt.figure(2)
plt.plot(tt, yy1, label='{f} Hz')
plt.title('Señal Senoidal desfasada en π/2 y amplificada a 3V')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.legend(title="Energía=")
plt.grid(True)
plt.show()

#------------------------------------------------------
#LA SEÑAL MODULADA POR OTRA FUNCIÓN
def funcion_seno_modulada(vc=1, dc=0, fs2=None, ph=0, nn=N, fm=fm):
    t=np.arange(0, nn) / fm
    sen=(vc*np.sin(2*np.pi*fs2*t+ph)+dc)
    return t,sen

fs2=1000
tt2, yy2 = funcion_seno_modulada(vc=1, dc=0, fs2=1000, ph=0, nn=N, fm=fm)

mody=yy2*yy

dt = 1/fm
energia = np.sum(yy2**2)*dt
muestras=len(yy2)
tiempoMuestras= dt
print(f'La energía de la función senoidal desfasada es: {energia:.4f}')
print(f'La cantidad de muestras es: {muestras}')
print(f'El tiempo entre muestras es: {tiempoMuestras}s')

plt.figure(3)
plt.plot(tt, mody)
plt.title('Señales moduladas')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.grid(True)
plt.show()

#----------------------------------------------------
#Señal recortada al 75%
vc=1
potencia=(vc**2)/2

#Calulo el 75% de la potencia
threshold=potencia*0.75

yyRecortada=np.clip(mody,-threshold,threshold)

dt = 1/fm
energia = np.sum(yyRecortada**2)*dt
muestras=len(yyRecortada)
tiempoMuestras= dt
print(f'La energía de la función senoidal desfasada es: {energia:.4f}')
print(f'La cantidad de muestras es: {muestras}')
print(f'El tiempo entre muestras es: {tiempoMuestras}s')

plt.figure(4)
plt.plot(tt, yyRecortada, label='{f} Hz')
plt.title('Señal recortada al 75% de su potencia en amplitud')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.legend(title="Frecuencias")
plt.grid(True)
plt.show()

#----------------------------------------
#Señal cuadrada
from scipy import signal #para hacer la señal cuadrada

# Parámetros
fs = 4000        # frecuencia de la señal (Hz)
T = 1/fs         # período (s)
t = np.arange(N)/fm

# Señal cuadrada
sq_wave = signal.square(2 * np.pi * fs * t)

dt = 1/fm
energia = np.sum(sq_wave**2)*dt
muestras=len(sq_wave)
tiempoMuestras= dt
print(f'La energía de la función cuadrada es: {energia:.4f}')
print(f'La cantidad de muestras es: {muestras}')
print(f'El tiempo entre muestras es: {tiempoMuestras}s')

# Graficar

plt.figure(5)
plt.plot(t, sq_wave, label="Señal cuadrada 4 kHz")
plt.title("Señal cuadrada de 4 kHz")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.grid(True)
plt.ylim(-1.5, 1.5)
plt.legend()
plt.show()

#----------------------------------------------
#pulso rectangular

# Parámetros
T = 0.02           # tiempo total de simulación [s]
A = 1.0           # amplitud del pulso
t0 = 0       # inicio del pulso [s]
Tp = 10e-3        # duración del pulso [s]

# Vector de tiempo
t = np.arange(0, T, 1/fm)

# Defino el pulso
pulso = np.zeros_like(t)                   # todo en cero
pulso[(t >= t0) & (t < t0 + Tp)] = A       # intervalo activo

dt = 1/fm
energia = np.sum(pulso**2)*dt
muestras=len(pulso)
tiempoMuestras= dt
print(f'La energía de la función senoidal desfasada es: {energia:.4f}')
print(f'La cantidad de muestras es: {muestras}')
print(f'El tiempo entre muestras es: {tiempoMuestras}s')

# Gráfico
plt.figure(figsize=(8,4))
plt.plot(t, pulso, lw=2)
plt.title("Pulso rectangular aislado")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.grid(True)
plt.show()

#----------------------------------------------
#Verificar ortogonalidad

tol = 1e-8

def verificar_ortogonalidad(y1, y2,senal1, senal2):
    if len(y1) != len(y2):
        raise ValueError("Las señales deben tener la misma longitud")
    # Producto interno discreto usando np.dot
    prodInterno=np.dot(y1,y2)*dt
    if abs(prodInterno)<tol:
        print(f'Las señales {senal1} y la {senal2} son ortogonales, con producto interno: {prodInterno}')
    else:
        print(f'Las señales {senal1} y la {senal2} no son ortogonales, su producto interno es: {prodInterno}')
    return 

# Caso 1: señal base vs desfazada
verificar_ortogonalidad(yy, yy1,'senoidal original','senoidal desfasada y amplificada')

# Caso 2: señal base vs modulada
verificar_ortogonalidad(yy, mody,'senoidal original','modulada')

# Caso 3: señal base vs recortada
verificar_ortogonalidad(yy, yyRecortada,'senoidal original','recortada')

# Caso 4: señal base vs señal cuadrada
min_len = min(len(yy), len(sq_wave))
verificar_ortogonalidad(yy[:min_len], sq_wave[:min_len],'senoidal original', 'cuadrada')


# Caso 5: señal base vs pulso rectangular
min_len = min(len(yy), len(pulso))
verificar_ortogonalidad(yy[:min_len], pulso[:min_len],'senoidal original','del pulso')


#-------------------------------------------------
# Autocorrelación

auto_yy = np.correlate(yy, yy, mode="full")*dt
lags_auto = np.arange(-len(yy)+1, len(yy))*dt
#print("El coeficiente de autocorrelación de la senoidal es: ",auto_yy)

# Correlaciones cruzadas
corr_y_y1 = np.correlate(yy, yy1, mode="full")*dt
lags_y1 = np.arange(-len(yy)+1, len(yy1))*dt
#print("El coeficiente de correlación entre la senoidal desfasada y la senoidal original es: ",corr_y_y1)

corr_y_mody = np.correlate(yy, mody, mode="full")*dt
lags_mody = np.arange(-len(yy)+1, len(mody))*dt
#print("El coeficiente de correlación entre la señal modulada y la senoidal es: ",corr_y_mody)

corr_y_rec = np.correlate(yy, yyRecortada, mode="full")*dt
lags_rec = np.arange(-len(yy)+1, len(yyRecortada))*dt
#print("El coeficiente de correlación entre la función recortada y la senoidal es: ",corr_y_rec)

corr_y_sq = np.correlate(yy, sq_wave, mode="full")*dt
lags_sq = np.arange(-len(yy)+1, len(sq_wave))*dt
#print("El coeficiente de correlación entre la señal cuadrada y la senoidal es: ",corr_y_sq)

corr_y_pulso = np.correlate(yy, pulso, mode="full")*dt
lags_pulso = np.arange(-len(yy)+1, len(pulso))*dt
#print("El coeficiente de correlación entre el pulso y la senoidal es: ",corr_y_pulso)

# Gráficos
plt.figure(figsize=(12,10))

plt.subplot(3,2,1)
plt.plot(lags_auto, auto_yy, label="Auto(yy)")
plt.title("Autocorrelación de la señal base")
plt.legend(); plt.grid()

plt.subplot(3,2,2)
plt.plot(lags_y1, corr_y_y1, label="Corr(yy, yy1)")
plt.title("Correlación con señal amplificada")
plt.legend(); plt.grid()

plt.subplot(3,2,3)
plt.plot(lags_mody, corr_y_mody, label="Corr(yy, mody)")
plt.title("Correlación con señal modulada")
plt.legend(); plt.grid()

plt.subplot(3,2,4)
plt.plot(lags_rec, corr_y_rec, label="Corr(yy, yyRecortada)")
plt.title("Correlación con señal recortada")
plt.legend(); plt.grid()

plt.subplot(3,2,5)
plt.plot(lags_sq, corr_y_sq, label="Corr(yy, sq_wave)")
plt.title("Correlación con señal cuadrada")
plt.legend(); plt.grid()

plt.subplot(3,2,6)
plt.plot(lags_pulso, corr_y_pulso, label="Corr(yy, pulso)")
plt.title("Correlación con pulso rectangular")
plt.legend(); plt.grid()

plt.tight_layout()
plt.show()

#------------------------------------------------
#IDENTIDAD TRIGONOMETRICA

def funcion_seno(vc=1, dc=0, fs=None, ph=0, nn=N, fm=fm):
    t=np.arange(0, nn) / fm
    sen=(vc*np.sin(2*np.pi*fs*t+ph)+dc)
    return t,sen

def funcion_coseno(vc=1, dc=0, fs=None, ph=0, nn=N, fm=fm):
    t=np.arange(0, nn) / fm
    cos=(vc*np.cos(2*np.pi*fs*t+ph)+dc)
    return t,cos


fs4=2000
tt4, yy4 = funcion_seno(vc=1, dc=0, fs=fs4, ph=0, nn=N, fm=fm)

fs5=1000
tt5, yy5 = funcion_seno(vc=1, dc=0, fs=fs5, ph=0, nn=N, fm=fm)

fs6= fs4-fs5
tt6, yy6 = funcion_coseno(vc=1, dc=0, fs=fs6, ph=0, nn=N, fm=fm)

fs7=fs4+fs5
tt7, yy7 = funcion_coseno(vc=1, dc=0, fs=fs7, ph=0, nn=N, fm=fm)


plt.plot(tt4, 2*yy4*yy5, label='Multiplicación de senoidales', linestyle='--')
plt.plot(tt,yy6-yy7 , label='Resta de cosenos', linestyle=':')
plt.title('Comprobación de identidad trigonometrica')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.legend(title="Frecuencias")
plt.grid(True)
plt.show()


#---------------------------------------------------
#Bonus
from scipy.io import wavfile
fm, data = wavfile.read("ruidos_olas.wav")  # fm = frecuencia de muestreo, data = array con muestras

print("Frecuencia de muestreo:", fm)
print("Forma del array de datos:", data.shape)

# Vector de tiempo
t = np.arange(len(data)) / fm

plt.figure(figsize=(10,4))
plt.plot(t, data)
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.title("Señal de audio WAV")
plt.grid()
plt.show()