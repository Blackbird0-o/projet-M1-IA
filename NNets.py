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

from dataProcessing import *
# -------------Scores and metrics-------------
def root_mean_squared_error(y_true, y_pred):
  '''
  Calculates RMSE
  Input : 
    y_true : numpy array, real labels
    y_pred : numpy array, predicted labels
  Output : 
    RMSE of y_pred - y_true
  '''

  return K.sqrt(K.mean(K.square(y_pred - y_true))) 
      
def getScores(real, result):
  '''
  Evaluates predictions of a model
  Input : 
    real : numpy array, real labels
    result : numpy array, predicted labels
  Output : 
    scoref1 : float, score f1 of given prediction
    modelError : float, model error
    confusion : numpy array, confusion matrix
  '''

  print('Precision :')
  print(precision_score(result, real))

  print('Recall :')
  print(recall_score(result, real))

  print('F1 Score :')
  scoref1 = f1_score(result, real)
  print(scoref1)

  print('MSE :')
  modelError = mean_squared_error(result, real)
  print(modelError)

  print('confusion_matrix : ')
  confusion = confusion_matrix(result, real)
  print(confusion)
  return scoref1, modelError, confusion

def getScores_cross(real, result, display=False):
  '''
  Evaluates predictions of a model
  Input : 
    real : numpy array, real labels
    result : numpy array, predicted labels
    display : parameter (Boolean), if True then prints all metrics value
  Output :
    confusion : numpy array, confusion matrix
  '''

  confusion = confusion_matrix(result, real)
  
  if display:
    print('Precision :')
    print(precision_score(result, real))
    print('Recall :')
    print(recall_score(result, real))
    print('F1 Score :')
    print(f1_score(result, real))
    print('MSE :')
    print('')
    print(mean_squared_error(result, real))
    print('confusion_matrix : ')
    print(confusion)
    print('')
  
  return confusion 

def recall(y_true, y_pred):
  '''
  Defines Recall metric.
  '''

  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
  return true_positives / (possible_positives + K.epsilon())

def precision(y_true, y_pred):
  ''' 
  Defines Precision metric.
  '''

  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
  return true_positives / (predicted_positives + K.epsilon())

def f1(y_true, y_pred):
  '''
  Defines f1 metric.
  '''

  preci = precision(y_true, y_pred)
  rec = recall(y_true, y_pred)
  return 2*((preci*rec)/(preci+rec+K.epsilon()))


# -------------Neural Nets-------------
def auto_encoder(X, X_tst):
  '''
  Defines, compiles and fits an autoencoder.
  Input : 
    X, X_tst : numpy arrays, train and test sets
  Output :
    X_autoencoded : numpy array, X transformedpredicted by model
    X_tst_autoencoded : numpy array, X transformedpredicted by model
    autoencoder : autoencoder model
  '''

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

  X_autoencoded = autoencoder.predict(X)
  X_tst_autoencoded = autoencoder.predict(X_tst)

  return X_autoencoded, X_tst_autoencoded, autoencoder

def auto_encoder_conv(X, X_tst):
  '''
  Defines, compiles and fits a convolutionnal autoencoder.
  Input : 
    X, X_tst : numpy arrays, train and test sets
  Output :
    X_autoencoded : numpy array, X transformedpredicted by model
    X_tst_autoencoded : numpy array, X transformedpredicted by model
    autoencoder : autoencoder model
  '''

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

  X_autoencoded = autoencoder.predict(X)
  X_tst_autoencoded = autoencoder.predict(X_tst)

  return X_autoencoded, X_tst_autoencoded, autoencoder

def maxinet(x_train,y_train,x_test,y_test):
  '''
  Defines, compiles and fits a Conv and LSTM Net
  Input :
    x_train,y_train,x_test,y_test : numpy arrays, train and test sets and labels
  Output : 
    model : neural net trained model
    np.rint(model.predict(x_test)) : int numpy array, label prediction of x_test
  '''

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

def maxinet_cross(x_train,y_train,x_test,y_test, tst=False):
  '''
  Defines, compiles and fits a Conv and LSTM Net. 
  Modified version of maxinet to run with cross validation.
  Input :
    x_train,y_train,x_test,y_test : numpy arrays, train and test sets and labels
  Output : 
    model : neural net trained model
    np.rint(model.predict(x_test)) : int numpy array, label prediction of x_test
  '''

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


# -------------Cross Val-------------
def cross_validation(X, y, splits=5, testing=False): 
  '''
  Runs cross validation on maxinet_cross Neural Net
  Input : 
    X, y : numpy arrays, dataset and labels
    splits : parameter (Int), number of splits of KFold cross validation
    testing : parameter (Boolean), if True shortens runtime for testing purpose
  Ouput : 
    scores : numpy array, confusion matrix from each split 
  '''

  # Separate exoplanet stars from non-exoplanet stars
  x_stars = X[np.where(y==0)]
  y_stars = y[np.where(y==0)]
  x_exo = X[np.where(y==1)]
  y_exo = y[np.where(y==1)]

  # Create splits
  kf = KFold(n_splits=splits, random_state=None, shuffle=False)
  split_stars = kf.split(x_stars)
  split_exo = kf.split(x_exo)
  scores = np.zeros((splits, 2, 2))

  for k in range(splits):
    # A bit of info
    print("Running split number ", k + 1)

    # Define splits
    spS = next(split_stars)
    spE = next(split_exo)
    idx_tra_S = spS[0]
    idx_tst_S = spS[1]
    idx_tra_E = spE[0]
    idx_tst_E = spE[1]

    # Create train and test sets
    x_tra = np.concatenate((x_stars[idx_tra_S], x_exo[idx_tra_E]))
    y_tra = np.concatenate((y_stars[idx_tra_S], y_exo[idx_tra_E]))
    x_tst = np.concatenate((x_stars[idx_tst_S], x_exo[idx_tst_E]))
    y_tst = np.concatenate((y_stars[idx_tst_S], y_exo[idx_tst_E]))

    # Shuffle datasets
    x_tra, y_tra = shuffle(x_tra, y_tra)
    x_tst, y_tst = shuffle(x_tst, y_tst)

    # Bootstrap datasets
    x_tra, y_tra = bootstrap(x_tra, y_tra)
    x_tst, y_tst = bootstrap(x_tst, y_tst, inv=False)

    # Scale datasets
    x_tra, x_tst = scale_datasets(x_tra, x_tst, param='RPN')

    # Run and evaluate NN
    pred = maxinet_cross(x_tra, y_tra, x_tst, y_tst, tst=testing)
    scores[k] = getScores_cross(y_tst, pred)

  return scores



# -------------Issue Data IDing-------------
def maxinet_ID_issue(x_train,y_train,x_test,y_test, tst=False):
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
                    batch_size=32,
                    validation_data=(x_test, y_test))
  
  return model

def cross_validation_ID_issue(X, y, splits=5, testing=False):
  # Separate exoplanet stars from non-exoplanet stars
  x_stars = X[np.where(y==0)]
  y_stars = y[np.where(y==0)]
  x_exo = X[np.where(y==1)]
  y_exo = y[np.where(y==1)]

  #scaling X
  X_scaled, bbb = scale_datasets(X, X, param='RPN')

  # Create splits
  kf = KFold(n_splits=splits, random_state=None, shuffle=False)
  split_stars = kf.split(x_stars)
  split_exo = kf.split(x_exo)
  scores = np.zeros((splits, 2, 2))

  ID_issues = np.empty(splits, dtype=object) 

  for k in range(splits):
    # A bit of info
    print("Running split number ", k + 1)

    # Define splits
    spS = next(split_stars)
    spE = next(split_exo)
    idx_tra_S = spS[0]
    idx_tst_S = spS[1]
    idx_tra_E = spE[0]
    idx_tst_E = spE[1]
    

    # Create train and test sets
    x_tra = np.concatenate((x_stars[idx_tra_S], x_exo[idx_tra_E]))
    y_tra = np.concatenate((y_stars[idx_tra_S], y_exo[idx_tra_E]))
    x_tst = np.concatenate((x_stars[idx_tst_S], x_exo[idx_tst_E]))
    y_tst = np.concatenate((y_stars[idx_tst_S], y_exo[idx_tst_E]))

    # Shuffle datasets
    x_tra, y_tra = shuffle(x_tra, y_tra)
    x_tst, y_tst = shuffle(x_tst, y_tst)

    # Bootstrap datasets
    x_tra, y_tra = bootstrap(x_tra, y_tra)
    x_tst, y_tst = bootstrap(x_tst, y_tst, inv=False)

    # Scale datasets
    x_tra, x_tst = scale_datasets(x_tra, x_tst, param='RPN')

    # Run and evaluate NN
    model = maxinet_ID_issue(x_tra, y_tra, x_tst, y_tst, tst=testing)

    #predict all data to identify which data is always missclassififed
    # Indeed we noticed that when those data were not in the train it caused 
    # problems when testing and vice-versa : even when traning with it, it would fail
    pred = np.rint(model.predict(X_scaled))
    scores[k] = getScores_cross(y, pred)
    pred = pred.flatten()
    ID_issues[k] = np.where((y != pred))[0]

  return ID_issues, scores