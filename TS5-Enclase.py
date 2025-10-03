# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 20:07:55 2025

@author: USUARIO
"""
import matplotlib.pyplot as plt

# =======================
# DATOS
# =======================

# Capacitor cerámico
freq_ceramico = [100, 120, 1000, 10000, 100000]
fase_ceramico = [-86.8, -86.8, -87.1, -86.6, -85.5]
logf_ceramico = [2, 2.079181246, 3, 4, 5]
logmod_ceramico = [1.192846115, 1.114677691, 3.209515015, 2.231316643, 1.283753383]

# Capacitor electrolítico
freq_electro = [100, 120, 1000, 10000, 100000]
fase_electro = [-87.2, -86.8, -72.1, -13.3, 36]
logf_electro = [2, 2.079181246, 3, 4, 5]
logmod_electro = [0.056904851, -0.024108864, -0.880744111, -1.364516253, -1.823908741]

# Bobina (cuatro vueltas)
freq_bobina = [100, 120, 1000, 10000]
logf_bobina = [2, 2.079181246, 3, 4]
l_bobina = [0.000093, 0.0001013, 0.0000986, 0.00009988]
r_bobina = [0.0112, 0.0107 , 0.0131, 0.1055 ]

# =======================
# FUNCION PARA GRAFICAR
# =======================
def graficar_componente( logf, logmod, fase):
    # Gráfico 1: log|Z| vs log f
    plt.figure(figsize=(7,5))
    plt.plot(logf, logmod, marker='o', linestyle='-', color="b")
    plt.xlabel("log10(Frecuencia) [log10(Hz)]")
    plt.ylabel("log10(|Z|) [log10(Ω)]")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    
    # Gráfico 2: fase vs log f
    plt.figure(figsize=(7,5))
    plt.plot(logf, fase, marker='o', linestyle='-', color="r")
    plt.xlabel("log10(Frecuencia) [log10(Hz)]")
    plt.ylabel("Fase [grados]")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    
def graficar_componente2( l, r, logf):
    # Gráfico 1: log|Z| vs log f
    plt.figure(figsize=(7,5))
    plt.plot(logf, l, marker='o', linestyle='-', color="b")
    plt.xlabel("log10(Frecuencia) [log10(Hz)]")
    plt.ylabel("L [µH]")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    
    # Gráfico 2: fase vs log f
    plt.figure(figsize=(7,5))
    plt.plot(logf, r, marker='o', linestyle='-', color="r")
    plt.xlabel("log10(Frecuencia) [log10(Hz)]")
    plt.ylabel("R [Ω]")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

# =======================
# LLAMADO PARA CADA COMPONENTE
# =======================
graficar_componente( logf_ceramico, logmod_ceramico, fase_ceramico)
graficar_componente( logf_electro, logmod_electro, fase_electro)
graficar_componente2(l_bobina, r_bobina, logf_bobina)

plt.show()
