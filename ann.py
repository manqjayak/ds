# Importing necessary libraries
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
# Load data from CSV file
data = pd.read_csv('StressLevelDataset.csv')  # Replace 'your_data.csv' with the actual file path

print(data.info())
print(data)
# Assuming your CSV has columns named 'feature' and 'target'
x = data.iloc[:,:-1]
y = data['stress_level'].values
print(x)
print(y)
# feature selection with corelation
# column_drop = ['self_esteem',
#                'sleep_quality',
#                'living_conditions',
#                'safety',
#                'basic_needs',
#                'academic_performance',
#                'teacher_student_relationship',
#                'social_support',
#                'headache']
# x = x.drop(columns = column_drop)

# Convert y to one-hot encoding for multiclass classification
y_onehot = to_categorical(y, num_classes=3)

print(f"len of dependent variable {len(x.columns.tolist())}")
# scaler = StandardScaler()
# X_standardized = scaler.fit_transform(x)
# Splitting the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(x, y_onehot, test_size=0.3, random_state=42)
print("--")
print(X_test)
print("--")


######################################################################
# Measure the training time
start_time = time.time()

# Creating a neural network model
model = Sequential()

# Adding the input layer
model.add(Dense(units=20, input_dim=X_train.shape[1], activation='relu'))  # 20 neurons in the first hidden layer

# # Adding N hidden layers
# for _ in range(8):
#     model.add(Dense(units=30, activation='relu'))  # 10 neurons in each hidden layer



# model.add(Dense(units=516, activation='relu'))  
# model.add(Dense(units=516, activation='relu'))  
model.add(Dense(units=324, activation='relu'))  
model.add(Dense(units=324, activation='relu'))  
model.add(Dense(units=256, activation='relu'))  
model.add(Dense(units=256, activation='relu'))  
model.add(Dense(units=128, activation='relu'))  
model.add(Dense(units=128, activation='relu'))  
model.add(Dense(units=64, activation='relu'))  
model.add(Dense(units=64, activation='relu'))  
# model.add(Dense(units=32, activation='relu'))  
# model.add(Dense(units=32, activation='relu'))  



# 90% 8 batch
# model.add(Dense(units=120, activation='relu'))  
# model.add(Dense(units=120, activation='relu'))  
# model.add(Dense(units=110, activation='relu'))  
# model.add(Dense(units=110, activation='relu'))  
# model.add(Dense(units=100, activation='relu'))  
# model.add(Dense(units=80, activation='relu'))  
# model.add(Dense(units=80, activation='relu'))  
# model.add(Dense(units=80, activation='relu'))  
# model.add(Dense(units=60, activation='relu'))  
# model.add(Dense(units=40, activation='relu'))  
# model.add(Dense(units=30, activation='relu'))  
# model.add(Dense(units=30, activation='relu'))  
# model.add(Dense(units=20, activation='relu'))  
# model.add(Dense(units=20, activation='relu'))  

# 89
# model.add(Dense(units=128, activation='relu'))  
# model.add(Dense(units=128, activation='relu'))  
# model.add(Dense(units=112, activation='relu'))  
# model.add(Dense(units=112, activation='relu'))  
# model.add(Dense(units=112, activation='relu'))  
# model.add(Dense(units=88, activation='relu'))  
# model.add(Dense(units=88, activation='relu'))  
# model.add(Dense(units=88, activation='relu'))  
# model.add(Dense(units=72, activation='relu'))  
# model.add(Dense(units=64, activation='relu'))  
# model.add(Dense(units=64, activation='relu'))  
# model.add(Dense(units=32, activation='relu'))  
# model.add(Dense(units=32, activation='relu'))  
# model.add(Dense(units=32, activation='relu'))  
# model.add(Dense(units=16, activation='relu'))  

# Adding the output layer
model.add(Dense(units=3, activation='softmax'))

# Compiling the model with mean squared error loss and Adam optimizer
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# Training the model on the training data
model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1)

# Calculate training time
training_time = time.time() - start_time
print("Training Time:", training_time, "seconds")

######################################################################
#evaluate

# Evaluating the model on the test data
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Making predictions with the trained model on the test data
predictions = model.predict(X_test)

# Convert predictions to one-hot encoding for comparison
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)

# Create a confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)

# Print the confusion matrix and accuracy
print("Confusion Matrix:")
print(conf_matrix)
print(f'Accuracy: {accuracy}')

####################################################################
# # single value data test
# data_test = data.iloc[1,:-1]

# print(data_test)
# data_test = data_test.values.reshape(1, -1)
# print(data_test)
# # Making a prediction for the new data point
# prediction = model.predict(data_test)

# # Convert prediction to class label
# predicted_class = np.argmax(prediction, axis=1)

# print("Raw Prediction:", prediction)
# print("Predicted Class:", predicted_class)