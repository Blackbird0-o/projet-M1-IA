#-------------Imports-------------
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier,AdaBoostClassifier

from mpl_toolkits.mplot3d import Axes3D 

from scipy.signal import savgol_filter

from imblearn.over_sampling import SMOTE
from imblearn.base import BaseSampler

from collections import Counter # counts the number of elements per class ({0: 5050, 1: 37})

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, BatchNormalization,CuDNNLSTM, LSTM,Conv1D,MaxPool1D,Permute,Reshape
from keras.optimizers import RMSprop, adam
from keras.utils import to_categorical

##########################################################################################################################
#Fonctions
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

def bootstrap(x_train,y_train) : 
    x_train1 = x_train[np.where(y_train == 1)[0]] #Separation du train_set selon le label
    x_train0 = x_train[np.where(y_train == 0)[0]]
    index_train = np.random.randint(0,x_train1.shape[0] , size=x_train0.shape[0]) #genere une liste d'index 
                                                                                  #aléatoire pour equilibrer les données
    x_train_1_boot = x_train1[index_train]
    y_train_boot = np.concatenate((np.ones(x_train0.shape[0]),np.zeros(x_train0.shape[0]))) #on génère une liste de labels avec autant de 1 que de 0
    x_train_boot = np.concatenate((x_train_1_boot,x_train0)) #on rassemble les données une fois équilibrées
    
    # shuffle
    index = np.arange(y_train_boot.shape[0])
    np.random.shuffle(index)
    x_train_boot = x_train_boot[index]
    y_train_boot = y_train_boot[index]
    
    return x_train_boot,y_train_boot

def dataload(path='data/') :
    # Loading datas
    data_train = pd.read_csv(path+'exoTrain.csv')
    data_test = pd.read_csv(path+'exoTest.csv')
    
    # transformation des label en array de 0 et 1
    y_train = np.array(data_train["LABEL"])-1
    y_test = np.array(data_test['LABEL'])-1
    
    # on charge les features
    x_train = np.array(data_train.drop('LABEL',axis=1))
    x_test = np.array(data_test.drop('LABEL',axis=1))
    
    return x_train,y_train,x_test,y_test

def pcaPlot(X, descr= 'temporel',plot_samples = 500):
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
      indexes = np.where(y_train_boot[0:plot_samples] == target)
      ax.scatter(x_PCA_plot[indexes,0]
                , x_PCA_plot[indexes,1],
                x_PCA_plot[indexes,2]
                , c = color
                , s = 50)
  ax.legend(['pas d\'exoplanetes', 'exoplanetes'])
  ax.grid()
  plt.show()
  return None

#Make an identity sampler
class FakeSampler(BaseSampler):

    _sampling_type = 'bypass'

    def _fit_resample(self, X, y):
        return X, y

def plot_resampling(X, y, sampling, ax):
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
  sampler = FakeSampler()

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
  plot_resampling(x_train, y_train, sampler, ax1)
  ax1.set_title('Original data - y={}'.format(Counter(y_train)))

  plot_resampling(x_train, y_train, SMOTE(random_state = 0), ax2)
  ax2.set_title('Resampling using {}'.format(SMOTE(random_state=0).__class__.__name__))
  fig.tight_layout()
  plt.show()
  return None

def getScores(pred, result): 
  
    print('Precision :') 
    print(precision_score(result, pred))

    print('Recall :') 
    print(recall_score(result, pred))

    print('F1 Score :') 
    scoref1 = f1_score(result, pred)
    print(scoref1) 
     
    print('MSE :') 
    modelError = mean_squared_error(result, pred)
    print(modelError) 
    

    print('') 
    print('confusion_matrix : ')
    confusion_matrix  = confusion_matrix(result, pred)
    print(confusion_matrix ) 
    print('')  
    return scoref1, modelError, confusion_matrix

    
def SVM(x_train,y_train,x_test,y_test) :
  
  x_train = np.abs(np.fft.fft(x_train))[0:,0:1000]
  x_test = np.abs(np.fft.fft(x_test))[0:,0:1000]
  
  x_train_boot,y_train_boot = bootstrap(x_train,y_train)
  x_train_sc, x_test_sc = scale_datasets(x_train, x_test, param='transpose',reshape=False)
  x_train_boot_sc, x_test_boot_sc = scale_datasets(x_train_boot, x_test_boot, param='transpose',reshape=False)

  
  
  clf_svc = SVC(gamma='auto', kernel='linear')
  
  model = clf_svc.fit(x_train_boot_sc, y_train_boot)
  
  return model.predict(x_test_sc)

def forest(x_train,y_train,x_test,y_test):
  
  x_train = np.abs(np.fft.fft(x_train))[0:,0:1000]
  x_test = np.abs(np.fft.fft(x_test))[0:,0:1000]
  
  x_train_boot,y_train_boot = bootstrap(x_train,y_train)
  x_train_sc, x_test_sc = scale_datasets(x_train, x_test, param='transpose',reshape=False)
  x_train_boot_sc, x_test_boot_sc = scale_datasets(x_train_boot, x_test_boot, param='transpose',reshape=False)

  
  clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
  
  model = clf.fit(x_train_boot_sc, y_train_boot)
  
  return model.predict(x_test_sc)

def maxiforest(x_train,y_train,x_test,y_test):
  
  x_train = np.abs(np.fft.fft(x_train))[0:,0:1000]
  x_test = np.abs(np.fft.fft(x_test))[0:,0:1000]
  
  x_train_boot,y_train_boot = bootstrap(x_train,y_train)
  x_train_sc, x_test_sc = scale_datasets(x_train, x_test, param='transpose',reshape=False)
  x_train_boot_sc, x_test_boot_sc = scale_datasets(x_train_boot, x_test_boot, param='transpose',reshape=False)

  
  clf = ExtraTreesClassifier(n_estimators=100, max_depth=2,random_state=0)
  
  model = clf.fit(x_train_boot_sc, y_train_boot)
  
  return model.predict(x_test_sc)

def Ada(x_train,y_train,x_test,y_test):
  
  x_train = np.abs(np.fft.fft(x_train))[0:,0:1000]
  x_test = np.abs(np.fft.fft(x_test))[0:,0:1000]
  
  x_train_boot,y_train_boot = bootstrap(x_train,y_train)
  x_train_sc, x_test_sc = scale_datasets(x_train, x_test, param='standardScaling',reshape=False)
  x_train_boot_sc, x_test_boot_sc = scale_datasets(x_train_boot, x_test_boot, param='standardScaling',reshape=False)

  
  clf = AdaBoostClassifier(n_estimators=100, random_state=0)
  
  model = clf.fit(x_train_boot_sc, y_train_boot)
  
  return model.predict(x_test_sc)


def net(X, y, X_tst, y_tst):
  '''
  Defines and fits a NN sequential model on X and y. It then tests the model with X_tst and y_tst
  '''

  # Specify model
  model = Sequential()
  model.add(Conv1D(filters=16, kernel_size=11, activation='relu', input_shape=X.shape[1:]))
  model.add(MaxPool1D(strides=4))
  model.add(BatchNormalization())
  
  model.add(Conv1D(filters=32, kernel_size=11, activation='relu'))
  model.add(MaxPool1D(strides=4))
  model.add(BatchNormalization())
  
  model.add(Conv1D(filters=64, kernel_size=11, activation='relu'))
  model.add(MaxPool1D(strides=4))
  model.add(BatchNormalization())
  
  model.add(Conv1D(filters=128, kernel_size=11, activation='relu'))
  model.add(MaxPool1D(strides=4))
  
  model.add(Flatten())
  model.add(Dropout(0.25))
  
  model.add(Dense(64,activation="relu"))
  
  #model.add(Permute((2,1)))
  
  model.add(Reshape((-1,1)))
  
  model.add(CuDNNLSTM(16, return_sequences=True))
  model.add(CuDNNLSTM(32, return_sequences=True))
  model.add(CuDNNLSTM(64, return_sequences=True))
  model.add(CuDNNLSTM(128))
  model.add(Dropout(0.25))
  
  model.add(Dense(32,activation="relu"))
  
  model.add(Dense(1, activation="sigmoid"))
  

  model.summary()
  model.compile(loss="binary_crossentropy",
                optimizer=adam(),
                metrics=["accuracy"])

  # Parameters
  batch_size = 32
  epochs = 60
  
  x_train, x_val, y_train, y_val = train_test_split(X, y, stratify=y,
                                                  test_size=0.3, random_state=123)

  # Perform fit
  history = model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=1,
                      shuffle=False,
                      validation_data=(x_val, y_val))


  # Print results
  score = model.evaluate(X_tst, y_tst, verbose=0)
  print('Test loss/accuracy: %g, %g' % (score[0], score[1]))
  
  plt.figure(figsize=(15, 5)) 
  # Plot history for accuracy
  plt.subplot(121)
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('model accuracy -- MLP')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  # summarize history for loss
  plt.subplot(122)
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss -- MLP')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.tight_layout()

  return model, model.predict(x=X_tst)


def scale_datasets(X_train, X_test, param='standardScaling', reshape=True):
    if param == 'standardScaling':
        if reshape:
            
            return StandardScaler().fit_transform(X_train).reshape(-1,3197,1), StandardScaler().fit_transform(X_test).reshape(-1,3197,1)
        else :
            return StandardScaler().fit_transform(X_train), StandardScaler().fit_transform(X_test)
    elif param == 'transpose':
        X_train = np.transpose(X_train)
        X_test = np.transpose(X_test)
        
        if reshape:
            return np.transpose(StandardScaler().fit_transform(X_train)).reshape(-1,3197,1), np.transpose(StandardScaler().fit_transform(X_test)).reshape(-1,3197,1)
        else :
            return np.transpose(StandardScaler().fit_transform(X_train)), np.transpose(StandardScaler().fit_transform(X_test))

######################################################################################
  
#-------------Cleaning up-------------
warnings.filterwarnings("ignore", category=FutureWarning)

plt.close('all')

x_train,y_train,x_test,y_test = dataload()

# création du vecteur temps (h)
t = np.arange(len(x_train[0])) * (36.0/60.0)
dt = 36* 60 # sampling rate (s) les données sont prises avec 36min d'écart
f = np.fft.fftfreq(x_train.shape[1],dt) # vecteur fréquence en (Hz)

#savgol filter
#x_train = savgol_filter(x_train,309,3) # cf script bruit.py pour parametres optimaux
#x_test = savgol_filter(x_test,309,3)

x_train_boot,y_train_boot = bootstrap(x_train,y_train)
x_test_boot,y_test_boot = bootstrap(x_test,y_test)

# Resampling with SMOTE (oversampling algorithm)
x_train_SMOTE, y_train_SMOTE = SMOTE(random_state=0).fit_resample(x_train, y_train)
x_test_SMOTE, y_test_SMOTE = SMOTE(random_state=0,k_neighbors=4).fit_resample(x_test, y_test)


# Scaling
x_train_sc, x_test_sc = scale_datasets(x_train, x_test, param='transpose',reshape=False)
x_train_boot_sc, x_test_boot_sc = scale_datasets(x_train_boot, x_test_boot, param='transpose',reshape=False)
x_train_Rsc, x_test_Rsc = scale_datasets(x_train, x_test)
x_train_boot_Rsc, x_test_boot_Rsc = scale_datasets(x_train_boot, x_test_boot)
x_train_SMOTE_sc, x_test_SMOTE_sc = scale_datasets(x_train_SMOTE, x_test_SMOTE)
x_train_T_sc, x_test_T_sc = scale_datasets(x_train, x_test, param= 'transpose')
x_train_boot_T_sc, x_test_boot_T_sc = scale_datasets(x_train_boot, x_test_boot, param= 'transpose')
x_train_SMOTE_sc, x_test_SMOTE_sc = scale_datasets(x_train_SMOTE, x_test_SMOTE, param= 'transpose')



#------------Classifiers on differently processed datasets-------------
prediction = maxiforest(x_train,y_train,x_test,y_test)
getScores(y_test, prediction)




#------------NN on different processed datasets-------------
#model, y_pred = net(x_train_boot_Rsc, y_train_boot, x_test_boot_Rsc, y_test_boot)
#model, y_pred = net(x_train_SMOTE_sc, y_train_SMOTE, x_test_SMOTE_sc, y_test_SMOTE)
#model, y_pred = net(x_train_Rsc, y_train, x_test_Rsc, y_test)

'''
model, y_pred = net(x_train_T_Rsc, y_train, x_test_T_Rsc, y_test)

predthr = np.where(y_pred > 0.5, 1, 0)
getScores(y_test, y_pred)
'''

