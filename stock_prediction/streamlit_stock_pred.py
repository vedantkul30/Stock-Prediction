
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,LSTM,Dense,Bidirectional,Dropout
from datetime import datetime,timedelta

st.header('Tesla Stock Market Predictor')
st.subheader('About the Project')
st.write("The model is trained on Tesla's stock prices from January 1, 2019, to January 1, 2024, covering a period of five years. Now, you can use the model to predict stock prices from January 1, 2024, onwards. Besides comparing price graphs from January 1, 2024, to your selected date, you can also predict the stock price for a specific date")
st.write("NOTE: THE LATEST DATE YOU CAN SELECT IS THE CURRENT DATE WHEN YOU RUN THE PROJECT.")
stock = st.text_input('Enter Stock Symbol', 'TSLA')
start = '2023-10-01'

end_date = st.date_input('Select End Date', datetime.today())

end_date = end_date + timedelta(days=1)
end = end_date.strftime('%Y-%m-%d')
print(end)
data = yf.download(stock, start,end)
data.reset_index(inplace = True)

print(data.shape)


st.subheader('Stock Data')
st.write(data)

data_test = pd.DataFrame(data.Close)
scaler = MinMaxScaler(feature_range=(0, 1))
data_test_scaled = scaler.fit_transform(data_test)

x = []
y = []

for i in range(50,data_test_scaled.shape[0]):
  x.append(data_test_scaled[i-50:i])
  y.append(data_test_scaled[i,0])

x,y = np.array(x),np.array(y)

input_layer = Input(shape=(x.shape[1], 1))
lstm1 = Bidirectional(LSTM(150, activation = 'relu' , return_sequences=True))(input_layer)
dropout1 = Dropout(0.1)(lstm1)
lstm2 = Bidirectional(LSTM(200, activation = 'relu' ))(dropout1)
output_layer = Dense(1,activation='relu')(lstm2)

model = Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer=Adam(learning_rate=0.003256984194582958),loss = 'mean_squared_error')
model_path = r'C:\Users\Vedant Kulkarni\PycharmProjects\stockpred\final_Stock_Prediction_Model.weights.h5'
model.load_weights(model_path)

y_predict = model.predict(x)
y_predict = y_predict.reshape(-1, 1)
y_predict = scaler.inverse_transform(y_predict)
y = scaler.inverse_transform(y.reshape(-1, 1))

print(len(y_predict))

st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(y_predict, 'r', label='Original Price')
plt.plot(y, 'g', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig4)

st.header('Compare the predicted and actual stock price for a particular date')
curr_date = st.date_input('Select the date whose price you want to know', datetime.today())
curr_date = datetime.combine(curr_date, datetime.min.time())

data['Date'] = pd.to_datetime(data['Date'])

matching_row = data[data['Date'] == curr_date]

if not matching_row.empty:
    row_number = matching_row.index[0]
    st.write(f'The actual stock value that day was: {data.Close.values[row_number]}')
    st.write(f'The predicted stock value that day is: {y_predict[row_number-50][0]}')
else:
    st.write(f'The date {curr_date} is not found in the dataset.')







