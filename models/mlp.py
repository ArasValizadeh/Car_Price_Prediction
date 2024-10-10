import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

data = pd.read_csv('data.csv')

categorial = data.select_dtypes(include=['object']).columns

label_encoder = LabelEncoder()

for col in categorial:
  print(col)
  data[col] = label_encoder.fit_transform(data[col])

data = data.dropna()

X = data.drop('MSRP', axis=1).values
y = data['MSRP'].values

print(X[:5])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train.shape[1])

model = keras.Sequential([
    keras.Input(shape=(15, )), # still not sure
    layers.Dense(180, activation="relu"), # seems like adding denser layers performs better than adding more layers, think it's a gradient issue then
    layers.Dropout(0.6),
    layers.Dense(180, activation="relu"),
    layers.Dropout(0.6),
    layers.Dense(180, activation="relu"),
    layers.Dropout(0.6),
    layers.Dense(1, name="output"),
])

model.compile(optimizer='adam', loss='mean_absolute_error') # kinda makes sense why larger values are predicted poorly considering smaller values

model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split=0.2)

loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

predictions = model.predict(X_test)

print(predictions[:5])
print("Real values: ", y_test[:5])