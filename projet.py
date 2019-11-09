#-------------Imports-------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm

#-------------Cleaning up-------------
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

#Bootstrapping
x_train1 = x_train[np.where(y_train == 1)[0]] #Separation du train_set selon le label
x_train0 = x_train[np.where(y_train == 0)[0]]
index_train = np.random.randint(0,x_train1.shape[0] , size=x_train0.shape[0]) #genere une liste d'index 
                                                                              #aléatoire pour equilibrer les données
x_train_1_boot = x_train1[index_train]
y_train_boot = np.concatenate((np.ones(x_train0.shape[0]),np.zeros(x_train0.shape[0]))) #on génère une liste de labels avec autant de 1 que de 0
x_train_boot = np.concatenate((x_train_1_boot,x_train0)) #on rassemble les données une fois équilibrées

#shuffle
index = np.arange(y_train_boot.shape[0])
np.random.shuffle(index)
x_train_boot = x_train_boot[index]
y_train_boot = y_train_boot[index]

# On passe dans l'espace de fourier
x_train_fft = np.abs(np.fft.fft(x_train_boot))
x_test_fft = np.abs(np.fft.fft(x_test))
# selection des 10 1ere harmoniques
train_feature = x_train_fft#[0:,0:10]
test_feature = x_test_fft#[0:,0:10]

# Scaling
x_train_sc = StandardScaler().fit_transform(x_train_boot)
x_train_fft_sc = StandardScaler().fit_transform(x_train_fft)

# MinMaxScaler
#IDEA : I think I should scale on the row dimension -> n.transpose
x_train_boot_T = np.transpose(x_train_boot)
x_train_mms = np.transpose(MinMaxScaler().fit_transform(x_train_boot_T))
x_train_fft_mms = MinMaxScaler().fit_transform(x_train_fft)


#-------------PCA------------- 
pca = PCA(n_components=10)
x_PCA = pca.fit_transform(x_train_sc)

# let's visualize the data in 3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_zlabel('Principal Component 3', fontsize = 15)
ax.set_title('ACP du signal temporel', fontsize = 20)
targets = [0,1]
colors = ['b', 'r']
plot_samples = 500
x_PCA_plot = x_PCA[0:plot_samples]

for target, color in zip(targets,colors):
    indexes = np.where(y_train_boot[0:plot_samples] == target)
    ax.scatter(x_PCA_plot[indexes,0]
               , x_PCA_plot[indexes,1],
               x_PCA_plot[indexes,2]
               , c = color
               , s = 50)
ax.legend(['pas d\'exoplanetes', 'exoplanetes'])
ax.grid()




#-------------PCA + MMS------------- 
pca_mms = PCA(n_components=10)
x_PCA_MMS = pca_mms.fit_transform(x_train_mms)

# let's visualize the data in 3d
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('ACP du signal temporel avec MMS', fontsize = 20)
targets = [0,1]
colors = ['b', 'r']
plot_samples = y_train_boot.shape[0]
x_PCA_MMS_plot = x_PCA_MMS[0:plot_samples]

for target, color in zip(targets,colors):
    indexes = np.where(y_train_boot[0:plot_samples] == target)
    ax.scatter(x_PCA_MMS_plot[indexes,0]
               , x_PCA_MMS_plot[indexes,1]
               , c = color
               , s = 50)
ax.legend(['pas d\'exoplanetes', 'exoplanetes'])
ax.grid()




#-------------PCA + FFT------------- 
pca_fft = PCA(n_components=10)
x_PCA_fft = pca_fft.fit_transform(x_train_fft_sc)

# let's visualize the data in 3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_zlabel('Principal Component 3', fontsize = 15)
ax.set_title('ACP du signal fréquentiel', fontsize = 20)
targets = [0,1]
colors = ['b', 'r']
plot_samples = 500
x_PCA_fft_plot = x_PCA_fft[0:plot_samples]

for target, color in zip(targets,colors):
    indexes = np.where(y_train_boot[0:plot_samples] == target)
    ax.scatter(x_PCA_fft_plot[indexes,0]
               , x_PCA_fft_plot[indexes,1],
               x_PCA_fft_plot[indexes,2]
               , c = color
               , s = 50)
ax.legend(['pas d\'exoplanetes', 'exoplanetes'])
ax.grid()

plt.show()