# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 21:21:19 2025

@author: USUARIO
"""
#Lo azul te da la energía, que es el valor más grande.

import numpy as np
import matplotlib.pyplot as plt
N=1000

xk=np.zeros(N,dtype=np.complex128())
#o se puede x=0j
#k=1 delta f es 1 y x(n), que sea senoidal
N=1000
fs=1
vc=1
ff=N
def funcion_seno(vc=None, dc=None, fs=None, ph=None, nn=N, ff=ff):
    t = np.arange(nn) / ff
    sen = vc * np.sin(2 * np.pi * fs * t + ph) + dc
    return t, sen


tt,xn= funcion_seno(vc=vc, dc=0, fs=fs, ph=0, nn=N, ff=ff)

for k in range (N):
    for n in range(N):
        xk [k] += xn[n]*np.exp(-1j*k*np.pi*2*n/N)