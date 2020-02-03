# projet-M1-IA

Project of Artificial Intelligence and Machine Learning. Course link : https://krzakala.github.io/ml_P6_2019_web/

The goal is to identify the stars that have exoplanets orbiting them based on the light flux collected by Kepler telescope during Camapaign 3. Kaggle project link : https://www.kaggle.com/keplersmachines/kepler-labelled-time-series-data and github of the creator : https://github.com/winterdelta/KeplerAI

We have one month to try and get the best results possible. The idea is not to win the kaggle project but simply to explore all the tools of ML and to face a real ML problem.

You can find our results and conclusion in rapport.pdf . Note that this report is in french as the course was taught in french.

## Architecture of the repository

Data is placed inside a 'data' folder which is ignored by git because it's too heavy (even for git large files treatment). The data is actually made of two datasets : exoTest.csv and exoTrain.csv (datasets can be downloaded here : https://www.kaggle.com/keplersmachines/kepler-labelled-time-series-data/download).

## Jupyter Notebooks

Jupyter notebooks are made to run directly in Google Colab. The only required modification to run it is : first to mount your drive and have the data inside it and second to specify the path (in the drive, which means it should start with 'drive/My Drive/") from which it should retrieve the data.

**That said, all jupyter notebooks were saved once ran so you do not need to run it once more to get a first look at the results.**

F_Nbook_w_crossval.ipynb is the final architecture we used. We added cross validation to better evaluate the model.

IDing_pbl_data.ipynb is a notebook where we identify the data that seems to be missclassified everytime during cross validation whether it is in the training set or not.

Template.ipynb is a template of the colab notebook structure we might need to test any neural net or autoencoder. It contains the following sections : drive mount, imports, data processing and metrics definition.

## Python files

All functions run from the file projet.py. This files imports the functions defined in the other files (such as classifier.py, NNets.py, ...) and then runs them if needed. This file is kind of the playground where we tried out things, data processing and such. 

dataProcessing.py contains all the functions for data processing prior to any classification or any training of a neural net. It also contains the functions used to build a PCA of the dataset.

classifiers.py contains all the different functions made to run a plethora of classifiers from sklearn.

NNets.py contains a few metrics definition to evaluate the models, some autoencoders as well as the most effective neural net we built : 'maxinet'.

visualization.py is a "library" we created to easily compare the raw data and the data that went through different transforms (fft, wavelet, etc).
