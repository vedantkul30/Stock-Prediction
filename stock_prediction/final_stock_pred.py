import keras_tuner as kt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

start = '2019-01-01'
end = '2024-01-01'

stock = 'TSLA'

data = yf.download(stock,start,end)

print(data)

split_index = int(len(data) * 0.80)
data_train = pd.DataFrame(data.Close.iloc[:split_index])
data_test_val = pd.DataFrame(data.Close.iloc[split_index:])

test_split_index = int(len(data_test_val) * 0.25)
data_test = pd.DataFrame(data_test_val[:test_split_index])
data_val = pd.DataFrame(data_test_val[test_split_index:])

print(len(data_val))

scaler = MinMaxScaler(feature_range= (0,1))
data_train_scale = scaler.fit_transform(data_train)
scale = scaler.scale_
print(scale)
data_val_scale = scaler.transform(data_val)
data_test_scale = scaler.transform(data_test)

X_train = []
y_train = []
X_val = []
y_val = []
X_test = []
y_test = []

for i in range(50, data_train_scale.shape[0]):
    X_train.append(data_train_scale[i - 50:i])
    y_train.append(data_train_scale[i, 0])

for i in range(50,data_val_scale.shape[0]):
  X_val.append(data_val_scale[i-50:i])
  y_val.append(data_val_scale[i,0])

for i in range(50,data_test_scale.shape[0]):
  X_test.append(data_test_scale[i-50:i])
  y_test.append(data_test_scale[i,0])

X_train,y_train = np.array(X_train),np.array(y_train)
X_val,y_val = np.array(X_val),np.array(y_val)
X_test,y_test = np.array(X_test),np.array(y_test)
def build_model(hp):

    input_layer = Input(shape=(X_train.shape[1], 1))

    num_layers = hp.Int('num_layers', min_value=2, max_value=5, step=1)

    x = input_layer
    for i in range(num_layers):
        units = hp.Int(f'units_{i}', min_value=50, max_value=200, step=50)
        activation = hp.Choice(f'activation_{i}', values=['relu', 'tanh'])
        dropout_rate = hp.Float(f'dropout_{i}', min_value=0.0, max_value=0.5, step=0.1)

        x = Bidirectional(LSTM(units, activation=activation, return_sequences=(i < num_layers - 1)))(x)
        x = Dropout(dropout_rate)(x)

    output_layer = Dense(1, activation='relu')(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG')

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')

    return model

tuner = kt.Hyperband(
    build_model,
    objective='val_loss',
    max_epochs=50,
    factor=3,
    directory='my_dir',
    project_name='lstm_tuning'
)


checkpoint_path = "final_Stock_Prediction_Model.weights.h5"
checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True, mode='min')

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

tuner.search(X_train, y_train, epochs=50, validation_data=(X_val, y_val),
             callbacks=[checkpoint_callback, early_stopping_callback])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print("Best Hyperparameters:")
print(best_hps.values)

best_model = tuner.get_best_models(num_models=1)[0]

print(best_model.summary())

best_model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val),
               callbacks=[checkpoint_callback, early_stopping_callback])

y_predict = best_model.predict(X_test)

scale = scaler.scale_

print(scale)

y_predict = y_predict*scale
y_test = y_test * scale

plt.figure(figsize=(10,8))
plt.plot(y_predict,'r',label = 'Predicted Price')
plt.plot(y_test,'g',label = 'Actual Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()







