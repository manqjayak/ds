# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

# Load data from CSV file
data = pd.read_csv('StressLevelDataset.csv')  # Replace 'your_data.csv' with the actual file path

# Assuming your CSV has columns for features and 'target' for the dependent variable
x = data.iloc[:,:-1]
y = data['stress_level'].values

# Splitting the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Creating an SVM model for multiclass classification
svm_model = SVC(random_state=42)
# svm_model = SVC(kernel='linear', C=1.0, random_state=42)

######------------

# # Define a parameter grid for GridSearchCV
# param_grid = {
#     'C': [0.1, 1, 10, 100],  # Regularization parameter
#     'kernel': ['linear', 'rbf'],  # Kernel type
#     'gamma': [0.01, 0.1, 1, 'scale']  # Kernel coefficient for 'rbf' kernel
# }

# # Create a GridSearchCV object
# grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=3, scoring='accuracy')

# # Training the SVM model
# # svm_model.fit(X_train, y_train)
# # Perform grid search to find the best hyperparameters
# grid_search.fit(X_train, y_train)

# # Get the best hyperparameters
# best_params = grid_search.best_params_

# # Making predictions with the trained model on the test data
# # predictions = svm_model.predict(X_test)

# # Use the best model for predictions
# best_model = grid_search.best_estimator_
# predictions = best_model.predict(X_test)

# # Create a confusion matrix
# conf_matrix = confusion_matrix(y_test, predictions)

# # Calculate accuracy
# accuracy = accuracy_score(y_test, predictions)

# # Print the confusion matrix and accuracy
# print("Confusion Matrix:")
# print(conf_matrix)
# print(f'Accuracy: {accuracy}')

######------------

# Set hyperparameter values
C_value = 1
kernel_value = 'rbf'  # You can use 'linear' or 'rbf' or other valid kernels
gamma_value = 'scale'  # You can use 'scale' or a numeric value

# Create an SVM model with specified hyperparameters
svm_model = SVC(C=C_value, kernel=kernel_value, gamma=gamma_value, random_state=42)

# Train the SVM model
svm_model.fit(X_train, y_train)

# Make predictions with the trained model on the test data
predictions = svm_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
# # Create a confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)

# Print the accuracy
print("Confusion Matrix:")
print(conf_matrix)
print(f'Accuracy: {accuracy}')

# single value data test
data_test = data.iloc[0,:-1].values.reshape(1, -1)

# Scale the new data point if you scaled your training data
# new_data_point_scaled = scaler.transform(new_data_point)

# Making a prediction for the new data point
prediction = svm_model.predict(data_test)

print("Predicted Class:", prediction)