# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 21:08:54 2025

@author: USUARIO
"""

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

N=8
n=np.arange(N)

def funcion_seno(vc=3, dc=4, fs=None, ph=0, nn=N):
    t=np.arange(0, nn)
    sen=(vc*np.sin(2*np.pi*fs*t+ph)+dc)
    return t,sen
tt, yy = funcion_seno(vc=1, dc=0, fs=4, ph=0, nn=N)



for n in range (0,7):
    y=yy[-n]
    n++



rxy= sig.correlate(x,y)
convxy=sig.convolve(x,y)

plt.figure(1)
plt.clf()
plt.plot(x, 'x:', label= 'x')
plt.plot(y, 'y:', label= 'y')
plt.plot(rxy, 'o:', label= 'rxy')
plt.legend()
plt.show()