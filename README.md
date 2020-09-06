# Time Series Anomaly Detection
Build LSTM autoencoder using the Keras API with Tensorflow 2 as the backend to detect anomalies (sudden price changes) in the S&P 500 index
#### Task 1: Import Libraries
#### Task 2: Load and Inspect the S&P 500 Index Data

1. Range: 1986->2018
2. Frequency: 'D'- Mon->Fri

![S$P Data](https://raw.githubusercontent.com/datablogger-ml/Anomaly-detection-with-Keras/master/Anomaly%20Detection%20Images/Data.png)

#### Task 3: Data Preprocessing
#### Task 4: Temporalize Data and Create Training and Test Splits
#### Task 5: Build an LSTM Autoencoder
#### Task 6: Train the Autoencoder
#### Task 7: Plot Metrics and Evaluate the Model
![Loss vs Epochs](https://raw.githubusercontent.com/datablogger-ml/Anomaly-detection-with-Keras/master/Anomaly%20Detection%20Images/Training_loss.png)
We have underfit the model as our val_loss<loss, we can change the parameters for a better fit.
![Training Loss](https://raw.githubusercontent.com/datablogger-ml/Anomaly-detection-with-Keras/master/Anomaly%20Detection%20Images/TrainingDIst.png)
With the help of the distribution plot of the training loss, we can observe that very few observations have an error > 0.65. If we set threshold = 0.65, any error > 0.65 on the test loss will be considered as an anomaly.
#### Task 8: Detect Anomalies in the S&P 500 Index Data

Plotting our threshold line at 0.65, all the loss values above it are anomalies

![Threshold](https://raw.githubusercontent.com/datablogger-ml/Anomaly-detection-with-Keras/master/Anomaly%20Detection%20Images/Threshold.png)

Depicting the anomalies with :red_circle: 

![S&P500 Anomalies](https://raw.githubusercontent.com/datablogger-ml/Anomaly-detection-with-Keras/master/Anomaly%20Detection%20Images/S%26P500_anomalies.png)
