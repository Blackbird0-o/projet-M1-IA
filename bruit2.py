#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import minimize
from tqdm import tqdm




def loss(x,s,s_fft) :
    k = 2*int(np.abs(x[0])) + 1     # k doit etre un entier positif impair
    n = int(np.abs(x[1]))
    
    if n < 2:
        n = 2
    if n > 4 :
        n = 4
    
    if k > 401 :             # k doit etre au max de la taille du vecteur s
        k = 401
    
    if k < 251 :
        k = 251
    
    if n >= k :
        n = k-1
    n = 3

    #print(k,n)
    s_filtered = savgol_filter(s,k,n)                    # signal lissé
    noise = s - s_filtered                               # bruit
    noise_fft = np.abs(np.fft.fft(noise))[0:s_fft.shape[0]]   # FFT du bruit
    
    w = np.arange(s_fft.shape[0]) +1       # on crée un vecteur poid pour moyenne pondérée

    w = w/sum(w)                 # on normalise

    return np.sum(np.abs(s_fft-noise_fft)*w)#*np.log(np.sum(np.abs(s-s_filtered)))


plt.close('all')

#-------------Preliminray Data Exploration-------------
# Loading datas
x_train,y_train,x_test,y_test = dataload()

x_train = x_test
y_train = y_test

# création du vecteur temps (h)
t = np.arange(len(x_train[0])) * (36.0/60.0)
dt = 36* 60 # sampling rate (s) les données sont prises avec 36min d'écart

f = np.fft.fftfreq(x_train.shape[1],dt)[0:1599] # vecteur fréquence en (Hz)

#-------------étude du bruit-------------
s = RPN(x_train)                                                # RPN du signal
s_fft = np.abs(np.fft.fft(s))[0:,0:1599]                        # FFT du signal

features = np.zeros((s.shape[0],2))
x_train_filtered = np.zeros(x_train.shape)

for i in tqdm(range(0,x_train_filtered.shape[0])) :
    
    x0 = [154,3]
    res = minimize(loss,x0,args=(s[i],s_fft[i]),tol=1e-4,method='Powell')
    x = res.x
    k = 2*int(np.abs(x[0])) + 1     # k doit etre un entier positif impair
    n = int(np.abs(x[1]))
    
    if n < 2:
        n = 2
    if n > 4 :
        n = 4
    
    if k > 401 :             # k doit etre au max de la taille du vecteur s
        k = 401 

    if k < 251 :
        k = 251
        
    if n >= k :
        n = k-1
        
    n = 3

    features[i] = np.array([k,n])
    x_train_filtered[i] = savgol_filter(s[i],k,n)  
    

x_features = np.concatenate((features,s_fft[0:,0:10]),axis=1)
x_features = np.concatenate((x_features,x_train_filtered),axis=1)

df = pd.DataFrame(x_features)

headers = ['Windows size','Polynom degrees']

for i in range(10) :
    headers.append('FFT '+str(i+1))

for i in range(x_train_filtered.shape[1]) :
    headers.append('signal filtred '+str(i+1))

df.columns = headers

df.to_csv(r'new_features_test.csv')




# lets visualize datas in 3d
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('windows size', fontsize = 15)
ax.set_ylabel('polynom degrees', fontsize = 15)
ax.set_title('titre', fontsize = 20)
targets = [0,1]
colors = ['b', 'r']
plot_samples =  features.shape[0]-1
x_PCA_fft_plot = features[0:plot_samples]

for target, color in zip(targets,colors):
    indexes = np.where(y_train[0:plot_samples] == target)
    ax.scatter(x_PCA_fft_plot[indexes,0]
               , x_PCA_fft_plot[indexes,1]
               , c = color)
ax.legend(['pas d\'exoplanetes', 'exoplanetes'])
ax.grid()
