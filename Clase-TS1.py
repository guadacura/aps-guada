# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 18:20:53 2025

@author: USUARIO
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal #para hacer la señal cuadrada
#sin(w0*t)*sin((wo/2*t))
#Lo que hay que hacer es multiplicarlos. Punto a punto porque son dos vectores.

#Señal recortada:Haces la potencia (A^2/2) y calculas si supera el 75% lo recortas.
#Te queda una meseta
#pones un límite de amplitud. El valor de la senoidal que tiene la potencia no tiene que 
#que superar el 75%

#Pulso rectangular: hay una funcion en una librería de munpy

ff=100000
N=500
fs=2000

def funcion_seno(vc=1, dc=0, fs=None, ph=0, nn=N, ff=ff):
    t=np.arange(0, nn) / ff
    sen=(vc*np.sin(2*np.pi*fs*t+ph)+dc)
    return t,sen

tt, yy = funcion_seno(vc=1, dc=0, fs=fs, ph=0, nn=N, ff=ff)

plt.figure(1)
plt.plot(tt, yy, label='{f} Hz')
plt.title('Señales Senoidales con Distintas Frecuencias')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.legend(title="Frecuencias")
plt.grid(True)
plt.show()

#LA FUNCION DEFASADA Y AMPLIFICADA EN PI/2
tt1, yy1 = funcion_seno(vc=3, dc=0, fs=fs, ph=np.pi/2, nn=N, ff=ff)

plt.figure(2)
plt.plot(tt1, yy1, label='{f} Hz')
plt.title('Señales Senoidales desfasada y amplificada en pi/2')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.legend(title="Frecuencias")
plt.grid(True)
plt.show()

#LA SEÑAL MODULADA POR OTRA FUNCIÓN
def funcion_seno_modulada(vc=1, dc=0, fs=None, ph=0, nn=N, ff=ff):
    t=np.arange(0, nn) / ff
    sen=(vc*np.sin(2*np.pi*fs*t+ph)+dc)
    return t,sen

tt2, yy2 = funcion_seno_modulada(vc=1, dc=0, fs=1000, ph=0, nn=N, ff=ff)

mody=yy2*yy

tt3, yy3= funcion_seno(vc=3, dc=0, fs=3/2*2000, ph=np.pi/2, nn=N, ff=ff)

plt.figure(3)
plt.plot(tt, mody)
plt.plot(tt2,yy2)
plt.plot(tt2,-yy2)
plt.title('Señal modulada')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.legend(title="Frecuencias")
plt.grid(True)
plt.show()


#Señal recortada al 75%
vc=1
potencia=(vc**2)/2

#Calulo el 75% de la potencia
threshold=potencia*0.75

yyRecortada=np.clip(yy,-threshold,threshold)

plt.figure(4)
plt.plot(tt, yyRecortada, label='{f} Hz')
plt.title('Señal recortada')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.legend(title="Frecuencias")
plt.grid(True)
plt.show()

#-----------------------------------------------------------------------------
#Señal cuadrada
        
# Parámetros
fs = 4000        # frecuencia de la señal (Hz)
T = 1/fs         # período (s)
fm = 100000     # frecuencia de muestreo (Hz) - mucho mayor a f
t = np.linspace(0, 5*T, int(fm*5*T), endpoint=False)  # 5 períodos

# Señal cuadrada
sq_wave = signal.square(2 * np.pi * fs * t)

# Graficar

plt.figure(5)
plt.plot(t*1e6, sq_wave, label="Señal cuadrada 4 kHz")  # t en microsegundos
plt.title("Señal cuadrada de 4 kHz")
plt.xlabel("Tiempo [µs]")
plt.ylabel("Amplitud")
plt.grid(True)
plt.ylim(-1.5, 1.5)
plt.legend()
plt.show()

#----------------------------------
#pulso rectangular

# Parámetros
fs = 100e3        # frecuencia de muestreo [Hz]
T = 1e-3          # tiempo total de simulación [s]
A = 1.0           # amplitud del pulso
t0 = 0       # inicio del pulso [s]
Tp = 10e-3        # duración del pulso [s]

# Vector de tiempo
t = np.arange(0, T, 1/fs)

# Defino el pulso
pulso = np.zeros_like(t)                   # todo en cero
pulso[(t >= t0) & (t < t0 + Tp)] = A       # intervalo activo

# Gráfico
plt.figure(figsize=(8,4))
plt.plot(t*1e3, pulso, lw=2)
plt.title("Pulso rectangular aislado")
plt.xlabel("Tiempo [ms]")
plt.ylabel("Amplitud")
plt.grid(True)
plt.show()

#--------------------------------------------------
#Ver ortogonalidad
# Señal senoidal base (ya definida como yy)
# Señal amplificada y desfasada (yy1)
# Señal modulada (mody)
# Señal recortada (yyRecortada)
# Señal cuadrada (sq_wave) --> ojo, tiene distinta longitud, hay que igualar
# Pulso rectangular (pulso) --> idem
tol=1e-8

def verificar_ortogonalidad(y1, y2):
  
    if len(y1) != len(y2):
        raise ValueError("Las señales deben tener la misma longitud")

    # Producto interno discreto
    prod = np.sum(y1 * y2)
    return prod




# Caso 1: señal base vs desfazada
prod=verificar_ortogonalidad(yy, yy1)
if abs(prod) < tol:
    print("Las señales del seno original y el seno desplazado son ortogonales (producto interno ≈ 0)")

else:
    print("Las señales del seno original y el seno desplazado NO son ortogonales (producto interno =", prod, ")")

# Caso 2: señal base vs modulada
prod=verificar_ortogonalidad(yy, mody)
if abs(prod) < tol:
    print("Las señales del seno original y la modulada son ortogonales (producto interno ≈ 0)")

else:
    print(" Las señales del seno original y la modulada NO son ortogonales (producto interno =", prod, ")")

# Caso 3: señal base vs recortada
prod=verificar_ortogonalidad(yy, yyRecortada)
if abs(prod) < tol:
    print(" Las señales del seno original y la recortada son ortogonales (producto interno ≈ 0)")

else:
    print(" Las señales del seno original y la recortada NO son ortogonales (producto interno =", prod, ")")

# Caso 4: señal base vs señal cuadrada
# --- ajusto tamaño para comparar ---
min_len = min(len(yy), len(sq_wave))
verificar_ortogonalidad(yy[:min_len], sq_wave[:min_len])
if abs(prod) < tol:
    print("Las señales del serno original y de la señal cuadrada son ortogonales (producto interno ≈ 0)")

else:
    print("Las señales del seno original y de la señal cuadrada NO son ortogonales (producto interno =", prod, ")")


# Caso 5: señal base vs pulso rectangular
min_len = min(len(yy), len(pulso))
verificar_ortogonalidad(yy[:min_len], pulso[:min_len])
if abs(prod) < tol:
    print("Las señales del seno original y el pulso son ortogonales (producto interno ≈ 0)")

else:
    print("Las señales del seno original y el pulso NO son ortogonales (producto interno =", prod, ")")

#------------------------------------------------------
# Autocorrelación de la señal base
auto_yy = signal.correlate(yy, yy, mode="full")

# Correlaciones cruzadas
corr_y_y1 = signal.correlate(yy, yy1, mode="full")
corr_y_mody = signal.correlate(yy, mody, mode="full")
corr_y_rec = signal.correlate(yy, yyRecortada, mode="full")
corr_y_sq = signal.correlate(yy, sq_wave, mode="full")
corr_y_pulso = signal.correlate(yy, pulso, mode="full")

lags = np.arange(-N+1, N)  # eje de retardos

# ==========================
# 4) Graficar
# ==========================
plt.figure(figsize=(12,10))

plt.subplot(3,2,1)
plt.plot(lags, auto_yy, label="Auto(yy)")
plt.title("Autocorrelación de la señal base")
plt.legend(); plt.grid()

plt.subplot(3,2,2)
plt.plot(lags, corr_y_y1, label="Corr(yy, yy1)")
plt.title("Correlación con señal amplificada")
plt.legend(); plt.grid()

plt.subplot(3,2,3)
plt.plot(lags, corr_y_mody, label="Corr(yy, mody)")
plt.title("Correlación con señal modulada")
plt.legend(); plt.grid()

plt.subplot(3,2,4)
plt.plot(lags, corr_y_rec, label="Corr(yy, yyRecortada)")
plt.title("Correlación con señal recortada")
plt.legend(); plt.grid()

plt.subplot(3,2,5)
plt.plot(lags, corr_y_sq, label="Corr(yy, sq_wave)")
plt.title("Correlación con señal cuadrada")
plt.legend(); plt.grid()

plt.subplot(3,2,6)
plt.plot(lags, corr_y_pulso, label="Corr(yy, pulso)")
plt.title("Correlación con pulso rectangular")
plt.legend(); plt.grid()

plt.tight_layout()
plt.show()

#-----------------------------
#IDENTIDAD TRIGONOMETRICA

def funcion_seno(vc=1, dc=0, fs=None, ph=0, nn=N, ff=ff):
    t=np.arange(0, nn) / ff
    sen=(vc*np.sin(2*np.pi*fs*t+ph)+dc)
    return t,sen

def funcion_coseno(vc=1, dc=0, fs=None, ph=0, nn=N, ff=ff):
    t=np.arange(0, nn) / ff
    cos=(vc*np.cos(2*np.pi*fs*t+ph)+dc)
    return t,cos


fs4=2000
tt4, yy4 = funcion_seno(vc=1, dc=0, fs=fs4, ph=0, nn=N, ff=ff)

fs5=1000
tt5, yy5 = funcion_seno(vc=1, dc=0, fs=fs5, ph=0, nn=N, ff=ff)

fs6= fs4-fs5
tt6, yy6 = funcion_coseno(vc=1, dc=0, fs=fs6, ph=0, nn=N, ff=ff)

fs7=fs4+fs5
tt7, yy7 = funcion_coseno(vc=1, dc=0, fs=fs7, ph=0, nn=N, ff=ff)


plt.plot(tt, yy4*yy5, label='Multiplicación de senoidales')
plt.plot(tt,yy6-yy7 , label='Resta de cosenos')
plt.title('Comprobación de identidad trigonometrica')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.legend(title="Frecuencias")
plt.grid(True)
plt.show()