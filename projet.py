# -------------Imports-------------
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from scipy.signal import savgol_filter

from NNets import *
from dataProcessing import *
from classifiers import *

######################################################################################
# -------------Cleaning up-------------
warnings.filterwarnings("ignore", category=FutureWarning)

plt.close('all')

x_train, y_train, x_test, y_test = dataload(merge=True)
'''
# création du vecteur temps (h)
t = np.arange(len(x_train[0])) * (36.0/60.0)
dt = 36 * 60  # sampling rate (s) les données sont prises avec 36min d'écart
f = np.fft.fftfreq(x_train.shape[1], dt)  # vecteur fréquence en (Hz)

x_train, y_train = bootstrap(x_train, y_train)
x_test, y_test = bootstrap(x_test, y_test, inv = False)
x_train, x_test = scale_datasets(x_train, x_test,param='RPN')

model, pred = maxinet(x_train,y_train,x_test,y_test)


#X_encoded, X_tst_encoded, autoencoder = auto_encoder(x_train, x_test)

getScores(y_test, pred)
'''
def RMSE(x_true,x_pred) :
  
  return np.sqrt(np.sum((x_true-x_pred)**2,axis = 1)) 

x_train, x_test = scale_datasets(x_train, x_test,param='RPN')

x_train0 = x_train[np.where(y_train ==0)]
x_train1 = x_train[np.where(y_train ==1)]
x_test0 = x_test[np.where(y_test ==0)]
x_test1 = x_test[np.where(y_test ==1)]

x_train0_encd, x_test0_encd, autoencoder = auto_encoder_conv(x_train0, x_test0)

x_train1_encd = autoencoder.predict(x_train1)
x_test1_encd = autoencoder.predict(x_test1)

x_train0RMSE = RMSE(x_train0,x_train0_encd)
x_train1RMSE = RMSE(x_train1,x_train1_encd)




'''
# for neural net reshape = True
x_train, x_test = scale_datasets(
    x_train, x_test, param='transpose', reshape=True)
x_train, y_train = bootstrap(x_train, y_train)

# savgol filter
# cf script bruit.py pour parametres optimaux
x_train = savgol_filter(x_train, 309, 3)
x_test = savgol_filter(x_test, 309, 3)

# fft
x_train = np.abs(np.fft.fft(x_train))[0:, 0:1000]
x_test = np.abs(np.fft.fft(x_test))[0:, 0:1000]

x_train = transform_dataset(x_train, nsamples=20)
x_test = transform_dataset(x_test, nsamples=20)

x_train, x_test = scale_datasets(x_train, x_test, reshape=True)
pcaPlot(x_train, y_train)

#x_train, x_test = scale_datasets(x_train, x_test,reshape=False)
pca = PCA(n_components=3)  # x_train.shape[1])
pca.fit(x_train)


x_train = pca.transform(x_train)
x_test = pca.transform(x_test)
'''
'''
x_train_boot,y_train_boot = bootstrap(x_train,y_train)
x_test_boot,y_test_boot = bootstrap(x_test,y_test)

# Resampling with SMOTE (oversampling algorithm)
x_train_SMOTE, y_train_SMOTE = SMOTE(random_state=0).fit_resample(x_train, y_train)
x_test_SMOTE, y_test_SMOTE = SMOTE(random_state=0,k_neighbors=4).fit_resample(x_test, y_test)


# Scaling
x_train_sc, x_test_sc = scale_datasets(x_train, x_test, param='transpose',reshape=True)
x_train_boot_sc, x_test_boot_sc = scale_datasets(x_train_boot, x_test_boot, param='transpose',reshape=True)
'''


# ------------Classifiers on differently processed datasets-------------
'''
print("knn")
prediction = knn(x_train,y_train,x_test,y_test)
getScores(y_test, prediction)

print('maxi trees')
prediction = maxiforest(x_train,y_train,x_test,y_test)
getScores(y_test, prediction)

#print('adaboost')
#prediction = Ada(x_train,y_train,x_test,y_test)
#getScores(y_test, prediction)
'''
'''
print('random forest')
prediction = SVM(x_train, y_train, x_test, y_test)
getScores(y_test, prediction)

# ------------NN on different processed datasets-------------
#model, y_pred = net(x_train_boot_Rsc, y_train_boot, x_test_boot_Rsc, y_test_boot)
#model, y_pred = net(x_train_SMOTE_sc, y_train_SMOTE, x_test_SMOTE_sc, y_test_SMOTE)
#model, y_pred = net(x_train_Rsc, y_train, x_test_Rsc, y_test)
'''
'''
N_model, N_y_pred = N_net(x_train_boot_sc, y_train_boot, x_test_boot_sc, y_test_boot)
predthr = np.where(N_y_pred > 0.5, 1, 0)
getScores(y_test_boot, predthr)
'''
