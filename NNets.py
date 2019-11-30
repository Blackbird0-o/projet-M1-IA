# -------------Imports-------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold

from keras import backend as K
from keras.models import Sequential 
from keras.models import  Model
from tensorflow.python.keras.layers import Activation, Dense, Dropout, Flatten, BatchNormalization, CuDNNLSTM, LSTM, Conv1D,UpSampling1D, MaxPool1D,MaxPooling1D, Permute, Reshape
from keras.optimizers import RMSprop, adam
from keras.utils import to_categorical


# -------------Scores and metrics-------------
def root_mean_squared_error(y_true, y_pred):
  return K.sqrt(K.mean(K.square(y_pred - y_true))) 
      
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
  confusion = confusion_matrix(result, pred)
  print(confusion)
  print('')
  return scoref1, modelError, confusion

def getScores_cross(pred, result, display=False):
  confusion = confusion_matrix(result, pred)
  
  if display:
    print('Precision :')
    print(precision_score(result, pred))
    print('Recall :')
    print(recall_score(result, pred))
    print('F1 Score :')
    print(f1_score(result, pred))
    print('MSE :')
    print('')
    print(mean_squared_error(result, pred))
    print('confusion_matrix : ')
    print(confusion)
    print('')
  
  return confusion 

def recall(y_true, y_pred):
  """Recall metric.
  Only computes a batch-wise average of recall.
  Computes the recall, a metric for multi-label classification of
  how many relevant items are selected.
  """
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
  return true_positives / (possible_positives + K.epsilon())

def precision(y_true, y_pred):
  """Precision metric.
  Only computes a batch-wise average of precision.
  Computes the precision, a metric for multi-label classification of
  how many selected items are relevant.
  """
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
  return true_positives / (predicted_positives + K.epsilon())

def f1(y_true, y_pred):
  preci = precision(y_true, y_pred)
  rec = recall(y_true, y_pred)
  return 2*((preci*rec)/(preci+rec+K.epsilon()))


# -------------Neural Nets-------------
def auto_encoder(X, X_tst):
  autoencoder = Sequential()

  # Encoder
  autoencoder.add(Conv1D(16, 10, activation='relu', padding='same', input_shape=X.shape[1:]))
  autoencoder.add(MaxPooling1D(4, padding='same'))
  autoencoder.add(Conv1D(8, 4, activation='relu', padding='same'))
  autoencoder.add(MaxPooling1D(4, padding='same'))
  #autoencoder.add(Conv1D(1, 4, activation='sigmoid', padding='same'))
  autoencoder.add(Reshape((200*8,)))
  autoencoder.add(Dense(200,activation = 'sigmoid'))

  # Decoder
  autoencoder.add(Dense(800,activation = 'relu'))
  autoencoder.add(Reshape((800,1)))
  autoencoder.add(Conv1D(8, 4, activation='relu', padding='same'))
  #autoencoder.add(UpSampling1D(2))
  autoencoder.add(Conv1D(8, 4, activation='relu', padding='same'))
  autoencoder.add(UpSampling1D(4))
  autoencoder.add(Conv1D(1, 4, activation='tanh'))

  autoencoder.summary()



  autoencoder.compile(optimizer='adam', loss = root_mean_squared_error)
  autoencoder.fit(X, X,epochs=10,batch_size=128)

  #encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('conv1d_3').output)
  #encoder.summary()

  X_encoded = autoencoder.predict(X)
  X_tst_encoded = autoencoder.predict(X_tst)

  return X_encoded, X_tst_encoded, autoencoder

def auto_encoder_conv(X, X_tst):
  autoencoder = Sequential()

  # Encoder
  autoencoder.add(Conv1D(16, 10, activation='relu', padding='same', input_shape=X.shape[1:]))
  autoencoder.add(MaxPooling1D(4, padding='same'))
  autoencoder.add(Conv1D(8, 4, activation='relu', padding='same'))
  autoencoder.add(MaxPooling1D(4, padding='same'))
  autoencoder.add(Conv1D(1, 4, activation='sigmoid', padding='same'))


  # Decoder
  autoencoder.add(Conv1D(8, 4, activation='relu', padding='same'))
  autoencoder.add(UpSampling1D(4))
  autoencoder.add(Conv1D(8, 4, activation='relu', padding='same'))
  autoencoder.add(UpSampling1D(4))
  autoencoder.add(Conv1D(1, 4, activation='tanh'))

  autoencoder.summary()



  autoencoder.compile(optimizer='adam', loss = root_mean_squared_error)
  autoencoder.fit(X, X,epochs=10,batch_size=128)

  #encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('conv1d_3').output)
  #encoder.summary()

  X_encoded = autoencoder.predict(X)
  X_tst_encoded = autoencoder.predict(X_tst)

  return X_encoded, X_tst_encoded, autoencoder

def maxinet(x_train,y_train,x_test,y_test):

  model = Sequential()

  model.add(Conv1D(16, 200, activation='relu', padding='same', input_shape=x_train.shape[1:]))
  model.add(MaxPooling1D(4, padding='same'))
  model.add(Conv1D(8, 100, activation='relu', padding='same'))
  model.add(MaxPooling1D(4, padding='same'))
  model.add(Conv1D(4, 10, activation='relu', padding='same'))
  model.add(Dropout(0.2))
  model.add(CuDNNLSTM(200, return_sequences=True))
  model.add(Dropout(0.2))
  model.add(CuDNNLSTM(20)) 
  model.add(Dropout(0.2))
  model.add(Dense(1, activation='sigmoid'))

  model.summary()

  model.compile(optimizer='adam', loss='binary_crossentropy',metrics=[precision]) #[f1, precision, "accuracy"]
  model.fit(x_train, y_train,
                  epochs=6,
                  shuffle = True,
                  batch_size=32)
  
  return model, np.rint(model.predict(x_test))


# -------------Cross Val-------------
def cross_validation(X, y, splits=5, testing=False): #PBL maybe because of bootstrap maybe do boostrap in here because we could have 30 times one exo in val data...
  #We first need to split the train set into exoplanet and non-exoplanet so that there isn't any expolanet in either the train or val test for example
  x_stars = X[np.where(y==0)]
  y_stars = y[np.where(y==0)]
  x_exo = X[np.where(y==1)]
  y_exo = y[np.where(y==1)]

  kf = KFold(n_splits=splits, random_state=None, shuffle=False)
  
  split_stars = kf.split(x_stars)
  split_exo = kf.split(x_exo)
  scores = np.zeros((splits, 2, 2))

  for k in range(splits):
    spS = next(split_stars)
    spE = next(split_exo)
    idx_tra_S = spS[0]
    idx_val_S = spS[1]
    idx_tra_E = spE[0]
    idx_val_E = spE[1]

    x_tra = np.concatenate((x_stars[idx_tra_S], x_exo[idx_tra_E]))
    y_tra = np.concatenate((y_stars[idx_tra_S], y_exo[idx_tra_E]))
    x_val = np.concatenate((x_stars[idx_val_S], x_exo[idx_val_E]))
    y_val = np.concatenate((y_stars[idx_val_S], y_exo[idx_val_E]))
    x_tra, y_tra = shuffle(x_tra, y_tra)
    x_val, y_val = shuffle(x_val, y_val)
    pred = maxinet_cross(x_tra, y_tra, x_val, y_val, tst=testing)
    scores[k] = getScores_cross(y_val, pred)

  return scores

def maxinet_cross(x_train,y_train,x_test,y_test, tst=False):
  model = Sequential()

  model.add(Conv1D(16, 200, activation='relu', padding='same', input_shape=x_train.shape[1:]))
  model.add(MaxPooling1D(4, padding='same'))
  model.add(Conv1D(8, 100, activation='relu', padding='same'))
  model.add(MaxPooling1D(4, padding='same'))
  model.add(Conv1D(4, 10, activation='relu', padding='same'))
  model.add(Dropout(0.2))
  model.add(CuDNNLSTM(200, return_sequences=True))
  model.add(Dropout(0.2))
  model.add(CuDNNLSTM(20)) 
  model.add(Dropout(0.2))
  model.add(Dense(1, activation='sigmoid'))

  model.compile(optimizer='adam', loss='binary_crossentropy',metrics=[precision]) #[f1, precision, "accuracy"]
  if tst:
    model.fit(x_train, y_train,
                    epochs=1,
                    shuffle = True,
                    batch_size=128)
  else:
    model.fit(x_train, y_train,
                    epochs=6,
                    shuffle = True,
                    batch_size=32)
  

  return np.rint(model.predict(x_test))