# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 18:55:21 2025

@author: USUARIO
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft

N=1000
fs=N
df=fs/N
ts=1/fs

def sen(ff,nn,amp=1, dc=0, ph=0, fs=2):
    N=np.arange(nn)
    t=N/fs
    x=dc+amp*np.sin(2*np.pi*ff*t+ph)
    return t,x

t1,x1= sen(ff=(N/4)*df, nn=N, fs=fs)
t2,x2= sen(ff=((N/4)+1)*df, nn=N, fs=fs)   
t3,x3= sen(ff=((N/4)+0.5)*df, nn=N, fs=fs)   

X1=(fft(x1))
X2=(fft(x2))
X3=fft(x3)

X1abs=np.abs(X1)
X2abs=np.abs(X2)
X3abs=np.abs(X3)

X1angle=np.angle(X1)
X2angle=np.angle(X2)
X3angle=np.angle(X3)


ff= np.arange(N)*df


plt.figure(1)
#plt.plot(ff,X1abs, 'x', label='X1 abs')
plt.plot(ff,np.log10(X1abs)*20,'x', label='X1 abs en V' )
#plt.plot(ff,X2abs, 'o', label='X2 abs')
plt.plot(ff,np.log10(X2abs)*20,'x', label='X2 abs en V' )
plt.plot(ff,np.log10(X3abs)*20,'x', label='X2 abs en V' )
plt.title('Comparación ...')
plt.xlabel('Frecuencia (Hz)')
plt.xlim([0,fs/2])
plt.ylabel('Amplitud [dB]')
plt.legend()
plt.grid()
plt.show ()




