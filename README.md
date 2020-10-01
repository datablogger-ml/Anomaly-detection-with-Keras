# Time Series Anomaly Detection

## Project Intro/Objective
Detecting Anomalies in the S&P 500 index using Tensorflow 2 Keras API with LSTM Autoencoder model.

### Methods Used
* Machine Learning / Deep Learning
* Time Series Analysis
* Anamoly Detection
* LSTM 
* Autoencoder

### Technologies
* python
* sklearn
* pandas, jupyter
* matplotlib, plotly
* tensorflow - keras api

## Project Description

### Data


### Methods

#### 1. Anamoly Detection
Anomaly detection is about identifying outliers in a time series data using mathematical models, correlating it with various influencing factors and delivering insights to business decision makers. Using anomaly detection across multiple variables and correlating it among them has significant benefits for any business. 

*Read this article to understand more on how [anomaly detection can help buinesses.](https://www.itproportal.com/features/five-reasons-why-anomaly-detection-is-important-for-your-ecommerce-business/)*

#### 2. LSTM

* <strong>WordCloud</strong> - 

* <strong>Topic Modelling</strong> - 

* <strong>Sentiment Analysis</strong> - 

#### 3. AutoEncoder


## Import Libraries
Import important libraries like pandas, numpy, matplotlib, plotly, tensorflow and sklearn.
## Load and Inspect the S&P 500 Index Data

1. Range: 1986->2018
2. Frequency: 'D'- Mon->Fri

![S$P Data](https://raw.githubusercontent.com/datablogger-ml/Anomaly-detection-with-Keras/master/Anomaly%20Detection%20Images/Data.png)

## Data Preprocessing
Split the dataset into 80% for the training set and remaining 20% for the test set.
## Temporalize Data and Create Training and Test Splits
The LSTM network takes the input in the form of subsequences of equal intervals of input shape (n_sample,n_timesteps,features).
## Build an LSTM Autoencoder
Model Summary :

Layer(type)  | Output Shape | # Param
------------ | -------------| --------
lstm (LSTM) | (None, 128) | 66560
dropout (Dropout) | (None, 128) | 0
repeat_vector (RepeatVector) | (None, 30, 128) | 0
lstm_1 (LSTM) | (None, 30, 128) | 131584
dropout_1 (Dropout) | (None, 30, 128) | 0
time_distributed (TimeDistributed) | (None, 30, 1) | 129

Total params: 198,273

Trainable params: 198,273

Non-trainable params: 0


## Train the Autoencoder
## Plot Metrics and Evaluate the Model
The metrics are saved inside the model variable, we can plot the training and validation loss wrt number of Epochs.
![Loss vs Epochs](https://raw.githubusercontent.com/datablogger-ml/Anomaly-detection-with-Keras/master/Anomaly%20Detection%20Images/Training_loss.png)

We have underfit the model as our val_loss<loss, we can change the parameters for a better fit.
![Training Loss](https://raw.githubusercontent.com/datablogger-ml/Anomaly-detection-with-Keras/master/Anomaly%20Detection%20Images/TrainingDIst.png)
With the help of the distribution plot of the training loss, we can observe that very few observations have an error > 0.65. If we set threshold = 0.65, any error > 0.65 on the test loss will be considered as an anomaly.
## Detect Anomalies in the S&P 500 Index Data

Plotting our threshold line at 0.65, all the loss values above it are anomalies

![Threshold](https://raw.githubusercontent.com/datablogger-ml/Anomaly-detection-with-Keras/master/Anomaly%20Detection%20Images/Threshold.png)

Depicting the anomalies with :red_circle: 

![S&P500 Anomalies](https://raw.githubusercontent.com/datablogger-ml/Anomaly-detection-with-Keras/master/Anomaly%20Detection%20Images/S%26P500_anomalies.png)
