import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pickle


# Set the CSV file path to load training data
data_training_file_path = "training_data/wine_quality.csv"

# Read the data into a DataFrame
df = pd.read_csv(data_training_file_path)

# Display sample data
print(df.head())

# Create the features (X) and target (y) sets
X = df.drop(columns=["quality"]).values
y = df["quality"].values

# Create training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Create the scaler instance
X_scaler = StandardScaler()

# Fit the scaler
X_scaler.fit(X_train)

# Scale the features data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

filename = 'scalers/model_scaler.sav'
pickle.dump(X_scaler, open(filename, 'wb'))

# Define the model - deep neural net with two hidden layers
number_input_features = 11
hidden_nodes_layer1 = 8
hidden_nodes_layer2 = 4

# Create a sequential neural network model
nn = Sequential()

# Add the first hidden layer
nn.add(Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu"))

# Add the second hidden layer
nn.add(Dense(units=hidden_nodes_layer2, activation="relu"))

# Add the output layer
nn.add(Dense(units=1, activation="linear"))

# Compile model
nn.compile(loss="mean_squared_error", optimizer="adam", metrics=["mse"])

# Fit the model
deep_net_model = nn.fit(X_train_scaled, y_train, epochs=100)

#evaluate (note the other example has more hidden layers but is just as effective so not used here)
# Evaluate Model 1 using testing data
model_loss, model_mse = nn.evaluate(X_test_scaled, y_test, verbose=2)

# Make predictions on the testing data
predictions = nn.predict(X_test_scaled).round().astype("int32")


# Create a DataFrame to compare the predictions with the actual values
results = pd.DataFrame({"predictions": predictions.ravel(), "actual": y_test})

# Display sample data
results.head(10)

# Set the model's file path
file_path = Path("models/wine_quality.h5")

# Export your model to an HDF5 file
nn.save(file_path)