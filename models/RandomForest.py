import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('/Users/arasvalizadeh/Desktop/Car_Price_Prediction/dataset/data.csv')

categorical = data.select_dtypes(include=['object']).columns

label_encoder = LabelEncoder()

for col in categorical:
    print(f'Encoding column: {col}')
    data[col] = label_encoder.fit_transform(data[col])

data = data.dropna()

X = data.drop('MSRP', axis=1).values
y = data['MSRP'].values

print('First 5 rows of features:\n', X[:5])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

model = RandomForestRegressor(
    n_estimators=400,  
    max_depth=None,   
    random_state=42
)


model.fit(X_train, y_train)


predictions = model.predict(X_test)


mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error on test set: {mae}')
print('Predictions for first 5 test samples:\n', predictions[:5])
print('Actual values for first 5 test samples:\n', y_test[:5])