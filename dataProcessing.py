#-------------Imports-------------
import numpy as np
import pywt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from mpl_toolkits.mplot3d import Axes3D

from imblearn.over_sampling import SMOTE
from imblearn.base import BaseSampler
from collections import Counter # counts the number of elements per class ({0: 5050, 1: 37})

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


#-------------Data Processing-------------
def RPN(x):
    '''
    Calcule la RPN d'un signal (Relative Power Noise)
    input :
        x = array numpy, le signal dont on souhaite calculer la RPN
        
    output :
        x_RPN = array numpy, la RPN du signal
    '''
    mean = np.mean(x,axis=1).reshape(x.shape[0],1)
    return (x-mean)/np.max(x,axis = 1).reshape(x.shape[0],1)
  
def shuffle(x,y):
    '''
    Shuffles the dataset along side the labels
    '''
    index = np.arange(y.shape[0])
    np.random.shuffle(index)
    x = x[index]
    y = y[index]
    
    return x,y
    
def bootstrap(x_train,y_train,inv=True) :
    '''
    Duplicates exoplanet stars until the number of exoplanet stars is identical to non-exoplanet stars
    Input : 
      x_train, y_train : numpy arrays dataset and associated labels
      inv : parameter (Boolean), if True then exoplanet stars flux signal are flipped (time symetry) and added to the dataset before bootstrap
    Output :
      x_train_boot, y_train_boot : numpy arrays, x_train, ytrain bootstraped
    '''

    if inv :
      x_train,y_train = inv_data(x_train,y_train)
      
    x_train1 = x_train[np.where(y_train == 1)[0]] #Separation du train_set selon le label
    x_train0 = x_train[np.where(y_train == 0)[0]]
    index_train = np.random.randint(0,x_train1.shape[0] , size=x_train0.shape[0]) #genere une liste d'index 
                                                                                  #aléatoire pour equilibrer les données
    x_train_1_boot = x_train1[index_train]
    y_train_boot = np.concatenate((np.ones(x_train0.shape[0]),np.zeros(x_train0.shape[0]))) #on génère une liste de labels avec autant de 1 que de 0
    x_train_boot = np.concatenate((x_train_1_boot,x_train0)) #on rassemble les données une fois équilibrées
    
    x_train_boot,y_train_boot  = shuffle(x_train_boot,y_train_boot)
    
    return x_train_boot,y_train_boot

def dataload(path='data/',merge=True) :
  '''
  Loads data and creates training and testing set and labels
  Input : 
    merge : parameter (Boolean), if true then the two original datasets are mixed together and shuffled
    path : parameter (String), sets the path from which to retrieve the datasets
  Output :
    x_train, y_train, x_test, y_test : numpy arrays, training and testing set adn labels
  '''

  # Loading datas
  data_train = pd.read_csv(path+'exoTrain.csv')
  data_test = pd.read_csv(path+'exoTest.csv')
  
  # transformation des label en array de 0 et 1
  y_train = np.array(data_train["LABEL"])-1
  y_test = np.array(data_test['LABEL'])-1
  
  # on charge les features
  x_train = np.array(data_train.drop('LABEL',axis=1))
  x_test = np.array(data_test.drop('LABEL',axis=1))
  
  if merge :
    data = np.concatenate((x_train,x_test))
    y = np.concatenate((y_train,y_test))
    data0 = data[np.where(y==0)[0]]
    y0 = y[np.where(y==0)[0]]
    data1 = data[np.where(y==1)[0]]
    y1 = y[np.where(y==1)[0]]
    
    x_train0,x_test0,y_train0,y_test0 = train_test_split(data0,y0, test_size = 0.1)
    x_train1,x_test1,y_train1,y_test1 = train_test_split(data1,y1, test_size = 0.1)
    
    x_train = np.concatenate((x_train0,x_train1))
    y_train = np.concatenate((y_train0,y_train1))
    x_test = np.concatenate((x_test0,x_test1))
    y_test = np.concatenate((y_test0,y_test1))
    
    x_train,y_train = shuffle(x_train,y_train)
    x_test,y_test = shuffle(x_test,y_test)
  
  return x_train,y_train,x_test,y_test

def pcaPlot(X, y, descr= 'temporel',plot_samples = 500):
  '''
  Defines and 10 components PCA of the dataset X and plots the first 3
  '''

  pca = PCA(n_components=10)
  x_PCA = pca.fit_transform(X)

  # let's visualize the data in 3d
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.set_xlabel('Principal Component 1', fontsize = 15)
  ax.set_ylabel('Principal Component 2', fontsize = 15)
  ax.set_zlabel('Principal Component 3', fontsize = 15)
  ax.set_title('ACP du signal ' + descr, fontsize = 20)
  targets = [0,1]
  colors = ['b', 'r']
  x_PCA_plot = x_PCA[0:plot_samples]

  for target, color in zip(targets,colors):
      indexes = np.where(y[0:plot_samples] == target)
      ax.scatter(x_PCA_plot[indexes,0]
                , x_PCA_plot[indexes,1],
                x_PCA_plot[indexes,2]
                , c = color
                , s = 50)
  ax.legend(['pas d\'exoplanetes', 'exoplanetes'])
  ax.grid()
  plt.show()
  return None

class FakeSampler(BaseSampler):
  '''
  Creates an identity sampler
  '''

  _sampling_type = 'bypass'
  def _fit_resample(self, X, y):
    return X, y

def plot_resampling(X, y, sampling, ax):
  '''
  Beautiful plot
  '''

  X_res, y_res = sampling.fit_resample(X, y)
  ax.scatter(X_res[:, 0], X_res[:, 1], c=y_res, alpha=0.8, edgecolor='k')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.get_xaxis().tick_bottom()
  ax.get_yaxis().tick_left()
  ax.spines['left'].set_position(('outward', 10))
  ax.spines['bottom'].set_position(('outward', 10))
  return Counter(y_res)

def SMOTE_plot(x_train, y_train):
  '''
  Plots both raw data and SMOTE generated data
  Input : 
    x_train, y_train : numpy arrays, training set and labels
  '''
  sampler = FakeSampler()

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
  plot_resampling(x_train, y_train, sampler, ax1)
  ax1.set_title('Original data - y={}'.format(Counter(y_train)))

  plot_resampling(x_train, y_train, SMOTE(random_state = 0), ax2)
  ax2.set_title('Resampling using {}'.format(SMOTE(random_state=0).__class__.__name__))
  fig.tight_layout()
  plt.show()
  return None

def transform_dataset(X, mode='wavelet', wname='db5',nsamples=10):
  '''
  Transforms a given dataset using a specified transformation
  '''

  if mode == 'wavelet':
    return pywt.dwt(X, wname)[0][:,0:nsamples]

  elif mode == 'fft':
    return np.abs(np.fft.fft(X))[:,0:nsamples]

  elif mode == 'all_in':
    allz = np.abs(np.fft.fft(X))[:,0:nsamples]
    wnames = ['db5','sym5','coif5','bior2.4']
    for wn in wnames:
      np.append(allz, pywt.dwt(X, wn)[0][:,0:nsamples], axis=1)
    return allz

def scale_datasets(X_train, X_test, param='standardScaling', reshape=True):
  '''
  Scales train and test sets using a specified method
  Input :
    x_train, x_test, numpy arrays
  Output : 
    two numpy arrays, x_train and x_test scaled and/or normalized
  '''

  SC = StandardScaler()
  train_shape = X_train.shape
  test_shape = X_test.shape
  
    
  if param == 'standardScaling':
    SC.fit(X_train)
    if reshape:
      return SC.transform(X_train).reshape(train_shape[0],train_shape[1],1), SC.transform(X_test).reshape(test_shape[0],test_shape[1],1)
    else :
      return SC.transform(X_train), SC.transform(X_test)
    
  elif param == 'transpose':
    X_train = np.transpose(X_train)
    if train_shape != test_shape :
      X_test = np.tile(X_test,(10,1))[0:train_shape[0]]
    X_test = np.transpose(X_test)
    SC.fit(X_train)
    if reshape:
      return np.transpose(SC.transform(X_train)).reshape(train_shape[0],train_shape[1],1), np.transpose(SC.transform(X_test))[0:test_shape[0]].reshape(test_shape[0],test_shape[1],1)
    else :
      return np.transpose(SC.transform(X_train)), np.transpose(SC.transform(X_test))[0:test_shape[0]]
  
  elif param == 'RPN':
    
    mean_train = np.mean(X_train,axis=1).reshape(X_train.shape[0],1) 
    mean_test = np.mean(X_test,axis=1).reshape(X_test.shape[0],1) # mean on each line which is allowed
    
    norm_train = np.max(np.abs(X_train),axis=1).reshape(-1,1)
    norm_test = np.max(np.abs(X_test),axis=1).reshape(-1,1) # max on each line which is also allowed
    
    if reshape:
      return ((X_train-mean_train)/norm_train) .reshape(train_shape[0],train_shape[1],1) , ((X_test-mean_test)/norm_test) .reshape(test_shape[0],test_shape[1],1)
    else :
      return ((X_train-mean_train)/norm_train)  , ((X_test-mean_test)/norm_test) 
  
  elif param == 'flatten':
    X_train = X_train.flatten().reshape((-1,1))
    X_test = X_test.flatten().reshape((-1,1))
    SC.fit(X_train)
    if reshape:
      return SC.transform(X_train).reshape(train_shape[0],train_shape[1],1), SC.transform(X_test).reshape(test_shape[0],test_shape[1],1)
    else :
      return SC.transform(X_train).reshape(train_shape[0],train_shape[1]), SC.transform(X_test).reshape(test_shape[0],test_shape[1])
  
  elif param == 'norm':
    norm_train = np.linalg.norm(X_train,axis=1).reshape(-1,1)
    norm_test = np.linalg.norm(X_test,axis=1).reshape(-1,1)
    if reshape:
      return (X_train/norm_train).reshape(train_shape[0],train_shape[1],1), (X_test/norm_test).reshape(test_shape[0],test_shape[1],1)
    else :
      return X_train/norm_train, X_test/norm_test
    
  elif param == 'norm_flatten':
    norm_train = np.linalg.norm(X_train)
    norm_test = np.linalg.norm(X_test,axis=1).reshape(-1,1)
    if reshape:
      return (X_train/norm_train).reshape(train_shape[0],train_shape[1],1), (X_test/norm_train).reshape(test_shape[0],test_shape[1],1)
    else :
      return X_train/norm_train, X_test/norm_train

def inv_data(X, y):
  '''
  Flips array when array corresponds to an exoplanet (time inversion)
  Input :
    X, y : numpy arrays, dataset and labels
  Output :
    two numpy arrays which correspond to : X + flipped data, y + labels of flipped data
  '''

  X_flipped = np.flip(X[np.where(y == 1)[0]], 1)
  y_flipped = np.ones((X_flipped.shape[0])) # all flipped arrays have a label of one by definition
  return np.concatenate((X, X_flipped)), np.concatenate((y, y_flipped))