# -------------Imports-------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, mean_squared_error
from sklearn.model_selection import train_test_split

from keras import backend as K
from tensorflow.keras.models import Sequential 
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


def maxinet(x_train,y_train,x_test,y_test,ep = 5, bs = 32):
  model = Sequential()

  model.add(Conv1D(16, 200, activation='relu', padding='same', input_shape=x_train.shape[1:]))
  model.add(MaxPooling1D(4, padding='same'))
  model.add(Conv1D(8, 100, activation='relu', padding='same'))
  model.add(MaxPooling1D(4, padding='same'))
  model.add(Conv1D(4, 10, activation='relu', padding='same'))
  model.add(Dropout(0.2))
  model.add(CuDNNLSTM(200, return_sequences=True))
  model.add(Dropout(0.2))
  model.add(CuDNNLSTM(70, return_sequences=True))
  model.add(Dropout(0.2))
  model.add(CuDNNLSTM(10))
  model.add(Dropout(0.2))
  model.add(Dense(1, activation='sigmoid'))

  model.summary()

  model.compile(optimizer='adam', loss='binary_crossentropy',metrics=[precision]) #[f1, precision, "accuracy"]
  model.fit(x_train, y_train,
                  epochs=ep,
                  batch_size=bs,
                  validation_data = (x_test,y_test))
  
  return model, np.rint(model.predict(x_test))