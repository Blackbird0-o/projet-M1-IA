#-------------Imports-------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter



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

f = np.fft.fftfreq(x_train.shape[1],dt) # vecteur fréquence en (Hz)



#-------------étude du bruit-------------
i = 9                                               # indice du signal étudié
s = x_train[i]                                       # signal
s_fft = np.abs(np.fft.fft(x_train[i]))               # FFT du signal
s_filtered = savgol_filter(x_train[i],301,3)         # signal lissé
s_filtered_fft = np.abs(np.fft.fft(s_filtered))  # FFT du signal lissé
noise = s - s_filtered                               # bruit
noise_fft = np.abs(np.fft.fft(noise))                # FFT du bruit

print('moyenne du bruit mesurée est de : ',np.mean(noise)) # on l'espère proche de 0

#-------------Plot-------------

plt.figure()
plt.plot(t,s,t,s_filtered)
plt.xlabel('temps (h)')
plt.ylabel('flus en e-/s')
plt.legend(['signal','signal lissé'])
plt.title('signaux temporel')


plt.figure()
plt.subplot(2,1,1)
plt.plot(t,noise)
plt.xlabel('temps (h)')
plt.ylabel('bruit en e-/s')
plt.legend(['bruit'])
plt.title('bruit temporel')

plt.subplot(2,1,2)
plt.loglog(f[0:1598],s_fft[0:1598],f[0:1598],noise_fft[0:1598],f[0:1598],s_filtered_fft[0:1598])
plt.xlabel('freq (Hz)')
plt.ylabel('bruit en e-/s')
plt.legend(['signal','bruit','signal lissé'])
plt.title('FFT')






