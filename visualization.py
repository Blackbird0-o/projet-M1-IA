# -------------Imports-------------
from dataProcessing import *
import warnings
import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


# -------------Cleaning up-------------
warnings.filterwarnings("ignore", category=FutureWarning)
plt.close('all')


# -------------Pretreatment-------------

# run in console so that you don't have a different set each time :
# ---
x_train, y_train, x_test, y_test = dataload(merge=True)
x_train, x_test = scale_datasets(
    x_train, x_test, param='transpose', reshape=False)
x_train, y_train = bootstrap(x_train, y_train)
# ---

# -------------Functons-------------
def comparaison_creator(X, svgON=False):
  '''
  Create several datasets from different transformations
  '''

  if svgON: 
    X = savgol_filter(X, 309, 3)
  
  X_fft = np.abs(np.fft.fft(X))
  X_wavelet = transform_dataset(X)
  X_fft_wavelet = transform_dataset(X_fft)

  return X_fft, X_wavelet, X_fft_wavelet

def comparaison_methods_plot(x1, x2, x3, x4, index):
  '''
  Compare visually the different transformations applied to a dataset
  '''

  fig, axs = plt.subplots(2, 2)

  axs[0, 0].plot(x1[index])
  axs[0, 0].set_title('raw scaled data')
  axs[0, 1].plot(x2[index], 'tab:orange')
  axs[0, 1].set_title('Fft')
  axs[1, 0].plot(x3[index], 'tab:green')
  axs[1, 0].set_title('Wavelet transform')
  axs[1, 1].plot(x4[index], 'tab:red')
  axs[1, 1].set_title('Fft and wavelet transform')

  plt.show()
  return None

def compare_stars(X, y, nsamples):
  '''
  Plots a few exoplanet nd non-exoplanet stars of a specific dataset
  '''

  index_exo = np.where(y == 1)
  index_non_exo = np.where(y == 0)

  X_exo = X[index_exo][:nsamples]
  X_non_exo = X[index_non_exo][:nsamples]

  for k in range(nsamples):
    plt.plot(X_exo[k], '--')
    plt.plot(X_non_exo[k], '-')
  
  plt.show()
  return None


# Run lines below to have a preview
x_train_fft, x_train_wv, x_train_fft_wv = comparaison(x_train)
compare_stars(x_train_fft_wv, y_train, 3)
