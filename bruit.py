#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#-------------Imports-------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import minimize

def RPN(x):
    '''
    Calcule la RPN d'un signal (Relative Power Noise)
    input :
        x = array numpy, le signal dont on souhaite calculer la RPN
        
    output :
        x_RPN = array numpy, la RPN du signal
        '''
    mean = np.mean(x,axis=1).reshape(x.shape[0],1)
    return (x-mean)/mean

def loss(k,s,s_fft) :
    k = 2*int(np.abs(k[0])) + 1     # k doit etre un entier positif impair
    
    if k > s.shape[1] :             # k doit etre au max de la taille du vecteur s
        k = s.shape[1]
    if k < 5 : 
        k = 5
    print(k)
    s_filtered = savgol_filter(s,k,3)                    # signal lissé
    noise = s - s_filtered                               # bruit
    noise_fft = np.abs(np.fft.fft(noise))[0:,0:s_fft.shape[1]]   # FFT du bruit
    
    #k = 2*int(np.abs(k)) + 1     # k doit etre un entier positif impair
    w = np.arange(s_fft.shape[1]) +1       # on crée un vecteur poid pour moyenne pondérée
    #w = 1/w
    w = w/sum(w)                 # on normalise
    w = np.tile(w,(s.shape[0],1))
    return np.sum(np.abs(s_fft-noise_fft)*w)

plt.close('all')

#-------------Preliminray Data Exploration-------------
# Loading datas
data_train = pd.read_csv('exoTrain.csv')
data_test = pd.read_csv('exoTest.csv')

# transformation des label en array de 0 et 1
y_train = np.array(data_train["LABEL"])-1
y_test = np.array(data_test['LABEL'])-1

# on charge les features
x_train = np.array(data_train.drop('LABEL',axis=1))
x_test = np.array(data_test.drop('LABEL',axis=1))

# création du vecteur temps (h)
t = np.arange(len(x_train[0])) * (36.0/60.0)
dt = 36* 60 # sampling rate (s) les données sont prises avec 36min d'écart

f = np.fft.fftfreq(x_train.shape[1],dt)[0:1599] # vecteur fréquence en (Hz)

#-------------étude du bruit-------------
s = RPN(x_train)                                                # RPN du signal
s_fft = np.abs(np.fft.fft(s))[0:,0:1599]                        # FFT du signal

# recherche des parametres optimaux pour le lissage :
x0 = np.array([154])
res = minimize(loss,x0,args=(s,s_fft),tol=1e-1,method='Powell')
#res = minimize(loss, x0, method='powell',args=(s,s_fft))#'powell')  

k_opti = 2*int(np.abs(res.x)) + 1 # 309

i = 30
s_filtered = savgol_filter(s,k_opti,3)               # signal lissé
s_filtered_fft = np.abs(np.fft.fft(s_filtered))[0:,0:1599]      # FFT du signal lissé
noise = s - s_filtered                               # bruit
noise_fft = np.abs(np.fft.fft(noise))[0:,0:1599]                # FFT du bruit 
 
#-------------Plot-------------

plt.figure()
plt.plot(t,s[i],t,s_filtered[i],'r')
plt.xlabel('temps (h)')
plt.ylabel('flus en e-/s')
plt.legend(['signal','signal lissé'])
plt.title('signaux temporel')

plt.figure()
plt.subplot(2,1,1)
plt.plot(t,noise[i])
plt.xlabel('temps (h)')
plt.ylabel('bruit en e-/s')
plt.legend(['bruit'])
plt.title('bruit temporel')

plt.subplot(2,1,2)
plt.loglog(f[0:1598],s_fft[i][0:1598],f[0:1598],noise_fft[i][0:1598],
           f[0:1598],s_filtered_fft[i][0:1598])
plt.xlabel('freq (Hz)')
plt.ylabel('bruit en e-/s')
plt.legend(['signal','bruit','signal lissé'])
plt.title('FFT')


plt.show()





