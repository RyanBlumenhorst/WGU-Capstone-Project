import pandas as pd
import numpy as np
import PySimpleGUI as sg
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Variables for all the readings
reading1 = None
reading2 = None
reading3 = None
reading4 = None
reading5 = None
reading6 = None
reading7 = None
reading8 = None
reading9 = None
reading10 = None
reading11 = None
reading12 = None
reading13 = None
reading14 = None
reading15 = None
reading16 = None
reading17 = None
reading18 = None
reading19 = None
reading20 = None
reading21 = None
reading22 = None

# Layout for the main window
layout = [[sg.Text("MDVP:Fo(Hz): ")], [sg.Input(key="reading1", do_not_clear=True)],
          [sg.Text("MDVP:Fhi(Hz):")], [sg.Input(key="reading2", do_not_clear=True)],
          [sg.Text("MDVP:Flo(Hz):")], [sg.Input(key="reading3", do_not_clear=True)],
          [sg.Text("MDVP: Jitter(%):")], [sg.Input(key="reading4", do_not_clear=True)],
          [sg.Text("MDVP: Jitter(Abs):")], [sg.Input(key="reading5", do_not_clear=True)],
          [sg.Text("MDVP:RAP:")], [sg.Input(key="reading6", do_not_clear=True)],
          [sg.Text("MDVP:PPQ:")], [sg.Input(key="reading7", do_not_clear=True)],
          [sg.Text("Jitter:DDP:")], [sg.Input(key="reading8", do_not_clear=True)],
          [sg.Text("MDVP:Shimmer:")], [sg.Input(key="reading9", do_not_clear=True)],
          [sg.Text("MDVP:Shimmer(dB):")], [sg.Input(key="reading10", do_not_clear=True)],
          [sg.Text("Shimmer:APQ3:")], [sg.Input(key="reading11", do_not_clear=True)],
          [sg.Text("Shimmer:APQ5:")], [sg.Input(key="reading12", do_not_clear=True)],
          [sg.Text("MDVP:APQ")], [sg.Input(key="reading13", do_not_clear=True)],
          [sg.Text("Shimmer:DDA:")], [sg.Input(key="reading14", do_not_clear=True)],
          [sg.Text("NHR:")], [sg.Input(key="reading15", do_not_clear=True)],
          [sg.Text("HNR:")], [sg.Input(key="reading16", do_not_clear=True)],
          [sg.Text("RPDE:")], [sg.Input(key="reading17", do_not_clear=True)],
          [sg.Text("DFA:")], [sg.Input(key="reading18", do_not_clear=True)],
          [sg.Text("Spread1:")], [sg.Input(key="reading19", do_not_clear=True)],
          [sg.Text("Spread2:")], [sg.Input(key="reading20", do_not_clear=True)],
          [sg.Text("D2:")], [sg.Input(key="reading21", do_not_clear=True)],
          [sg.Text("PPE")], [sg.Input(key="reading22", do_not_clear=True)],
          [sg.Button("Submit")]]

# Show Main Window
window = sg.Window("Parkinson's Disease Detection Program", layout)

# Main window loop
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break
    elif event == 'Submit':
        for i in values:
            if values[i] == '' or values[i] is None:
                sg.Window("Parkinson's Disease Detection Program", layout=[[sg.Text("Invalid or Missing Data")], [sg.Button("OK")]]).read()
                break

        reading1 = float(values["reading1"])
        reading2 = float(values["reading2"])
        reading3 = float(values["reading3"])
        reading4 = float(values["reading4"])
        reading5 = float(values["reading5"])
        reading6 = float(values["reading6"])
        reading7 = float(values["reading7"])
        reading8 = float(values["reading8"])
        reading9 = float(values["reading9"])
        reading10 = float(values["reading10"])
        reading11 = float(values["reading11"])
        reading12 = float(values["reading12"])
        reading13 = float(values["reading13"])
        reading14 = float(values["reading14"])
        reading15 = float(values["reading15"])
        reading16 = float(values["reading16"])
        reading17 = float(values["reading17"])
        reading18 = float(values["reading18"])
        reading19 = float(values["reading19"])
        reading20 = float(values["reading20"])
        reading21 = float(values["reading21"])
        reading22 = float(values["reading22"])

        break

# Create input data from main window
input_data = (reading1, reading2, reading3, reading4, reading5, reading6, reading7, reading8, reading9, reading10, reading11,
              reading12, reading13, reading14, reading15, reading16, reading17, reading18, reading19, reading20, reading21,
              reading22)


sg.Window(title="Parkinson's Disease Detection Program", layout=[[sg.Text("Input Data Has successfully been read")], [sg.Button("OK")]], margins=(100, 50)).read()


# Read The Data
df = pd.read_csv('parkinsons.data')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df.head()

print(df)

# Get the features and labels
features = df.loc[:, df.columns != 'status'].values[:, 1:]
labels = df.loc[:, 'status'].values

# Scale the features to between -1 and 1
scaler = MinMaxScaler((-1, 1))
x = scaler.fit_transform(features)
y = labels

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

# Train the model
model = XGBClassifier()
model.fit(x_train, y_train)

# Calculate the accuracy
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred) * 100
acc = round(acc, 2)

# Scale input data
input_data_np = np.asarray(input_data)
input_data_re = input_data_np.reshape(1, -1)
s_data = scaler.transform(input_data_re)

# Prediction, 1 for positive test, 0 for negative test
input_data_pred = model.predict(s_data)

# Plot input data against a positive test result
plt.plot(x[0])
plt.plot(s_data[0])
plt.legend(['Test Data', 'Input Data'])
plt.show()

# Show results
if input_data_pred[0] == 0:
    sg.Window(title="Results", layout=[[sg.Text("The test has came back negative, there was no Parkinson\'s Disease detected, Accuracy = " + str(acc) + "%")], [sg.Button("OK")]], margins=(100, 50)).read()
else:
    sg.Window(title="Results", layout=[[sg.Text("The test has came back positive, Parkinson\'s Disease has been detected, Accuracy = " + str(acc) + "%")], [sg.Button("OK")]], margins=(100, 50)).read()