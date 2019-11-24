# -------------Imports-------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, mean_squared_error
from sklearn.model_selection import train_test_split

from keras import backend as K

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, BatchNormalization, CuDNNLSTM, LSTM, Conv1D,UpSampling1D, MaxPool1D,MaxPooling1D, Permute, Reshape
from keras.optimizers import RMSprop, adam
from keras.utils import to_categorical


# -------------Scores and metrics-------------
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
def net(X, y, X_tst, y_tst):
  '''
  Defines and fits a NN sequential model on X and y. It then tests the model with X_tst and y_tst
  '''

  # Specify model
  model = Sequential()
  model.add(Conv1D(filters=16, kernel_size=11,
                    activation='relu', input_shape=X.shape[1:]))
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

  model.add(Dense(64, activation="relu"))

  # model.add(Permute((2,1)))

  model.add(Reshape((-1, 1)))

  model.add(CuDNNLSTM(16, return_sequences=True))
  model.add(CuDNNLSTM(32, return_sequences=True))
  model.add(CuDNNLSTM(64, return_sequences=True))
  model.add(CuDNNLSTM(128))
  model.add(Dropout(0.25))

  model.add(Dense(32, activation="relu"))

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

def N_net(X, y, X_tst, y_tst):
  '''
  Defines and fits a NN sequential model on X and y. It then tests the model with X_tst and y_tst
  '''

  # Specify model
  model = Sequential()

  model.add(Conv1D(filters=16, kernel_size=11,
                    activation='softsign', input_shape=X.shape[1:]))
  model.add(MaxPool1D(strides=4))
  #model.add(BatchNormalization())

  model.add(Flatten())
  model.add(Dropout(0.45))

  model.add(Dense(300, activation="relu"))

  # model.add(Permute((2,1)))

  model.add(Reshape((-1, 1)))
  '''
  model.add(Reshape((-1,1,1,1)))
  
  model.add(ConvLSTM2D(16, 1, return_sequences=True))
  model.add(ConvLSTM2D(32, 1, return_sequences=True))
  model.add(ConvLSTM2D(64, 1, return_sequences=True))
  model.add(ConvLSTM2D(128, 1))
  model.add(Reshape((-1,)))
  '''
  model.add(CuDNNLSTM(16, return_sequences=True))
  model.add(CuDNNLSTM(32, return_sequences=True))
  model.add(CuDNNLSTM(64, return_sequences=True))
  model.add(CuDNNLSTM(128))
  model.add(Dropout(0.45))

  model.add(Dense(32, activation="relu"))

  model.add(Dense(1, activation="sigmoid"))

  model.summary()
  model.compile(loss="binary_crossentropy",
                optimizer=adam(),
                metrics=[f1, precision, "accuracy"])

  # Parameters
  batch_size = 250
  epochs = 20
  '''
  ##with x_val
  x_train, x_val, y_train, y_val = train_test_split(X, y, stratify=y,
                                                  test_size=0.3, random_state=123)
  # Perform fit
  history = model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=1,
                      shuffle=False,
                      validation_data=(x_val, y_val))
  '''
  # without x_val
  # Perform fit
  history = model.fit(X, y,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=1,
                      shuffle=True,
                      validation_data=(X_tst, y_tst))

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

def auto_encoder(X, X_tst):
  #X = X.reshape(-1,1)
  #X_tst = X_tst.reshape(-1,1)
  autoencoder = Sequential()

  # Encoder
  autoencoder.add(Conv1D(16, 10, activation='relu', padding='same', input_shape=X.shape[1:]))
  autoencoder.add(MaxPooling1D(4, padding='same'))
  autoencoder.add(Conv1D(8, 4, activation='relu', padding='same'))
  autoencoder.add(MaxPooling1D(4, padding='same'))
  autoencoder.add(Conv1D(8, 4, activation='relu', padding='same'))

  # Decoder
  autoencoder.add(Conv1D(8, 4, activation='relu', padding='same'))
  autoencoder.add(UpSampling1D(4))
  autoencoder.add(Conv1D(8, 4, activation='relu', padding='same'))
  autoencoder.add(UpSampling1D(4))
  autoencoder.add(Conv1D(1, 4, activation='relu'))
  
  '''
  autoencoder.add(UpSampling1D(4))
  autoencoder.add(Conv1D(1, 4, activation='sigmoid', padding='same'))
  '''
  autoencoder.summary()



  autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
  autoencoder.fit(X, X,
                  epochs=50,
                  batch_size=128,
                  validation_data=(X_tst, X_tst))

  encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('conv1d_3').output)
  encoder.summary()

  X_encoded = encoder.predict(X)
  X_tst_encoded = encoder.predict(X_tst)

  return X_encoded, X_tst_encoded