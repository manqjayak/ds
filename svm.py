# Importing necessary libraries
import time
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif

# Load data from CSV file
data = pd.read_csv('StressLevelDataset.csv')  # Replace 'your_data.csv' with the actual file path

# Assuming your CSV has columns for features and 'target' for the dependent variable
x = data.iloc[:,:-1]
y = data['stress_level'].values

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


# Calculate the correlation matrix
correlation_matrix = data.corr()

# Print the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)

# # Plot a heatmap of the correlation matrix
# plt.figure(figsize=(8, 6))
# plt.imshow(correlation_matrix, cmap='viridis', interpolation='none', aspect='auto')
# plt.colorbar()
# plt.xticks(range(len(correlation_matrix)), correlation_matrix.columns, rotation='vertical')
# plt.yticks(range(len(correlation_matrix)), correlation_matrix.columns)
# plt.title('Correlation Heatmap')
# plt.show()

#################################################################################

# Splitting the data into training and testing sets (75% train, 25% test)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)



######################################################

# # Create an LDA model
# lda = LinearDiscriminantAnalysis(
#     solver='eigen',           # 'svd', 'lsqr', 'eigen'
#     shrinkage=0.5,         # float between 0 and 1 or 'auto'
#     priors=None,            # array-like, class priors
#     n_components=2       # int, number of components
# )

####################################################
# lda.fit(X_train, y_train)

# # Transform the training and testing data using LDA
# X_train_lda = lda.transform(X_train)
# X_test_lda = lda.transform(X_test)



# Apply PCA for dimensionality reduction
# pca = PCA(n_components=2)  # Adjust the number of components based on your requirements
# X_train_pca = pca.fit_transform(X_train)
# X_test_pca = pca.transform(X_test)


# Create a SelectKBest object with the f_classif, chi2, and mutual_info_classif scoring function
k_best = SelectKBest(score_func=chi2, k=9)  # Select the top 2 features

# Fit and transform the training data
X_train= k_best.fit_transform(X_train, y_train)

# Transform the test data
X_test = k_best.transform(X_test)


#######################################################
#model SVM

# Creating an SVM model for multiclass classification
svm_model = SVC(random_state=42)
# svm_model = SVC(kernel='linear', C=1.0, random_state=42)

# Set hyperparameter values
C_value = 1
kernel_value = 'rbf'  # You can use 'linear' or 'rbf' or other valid kernels
gamma_value = 'scale'  # You can use 'scale' or a numeric value

# Create an SVM model with specified hyperparameters
# param_grid = {
#     'C': [0.1, 1, 10, 100],  # Regularization parameter
#     'kernel': ['linear', 'rbf'],  # Kernel type
#     'gamma': [0.01, 0.1, 1, 'scale']  # Kernel coefficient for 'rbf' kernel
# }

# Measure the training time
start_time = time.time()

svm_model = SVC(C=C_value, kernel=kernel_value, gamma=gamma_value, random_state=42)

# Train the SVM model
svm_model.fit(X_train, y_train)

# Calculate training time
training_time = time.time() - start_time
print("Training Time:", training_time, "seconds")

# Make predictions with the trained model on the test data
predictions = svm_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
# # Create a confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)

# Print the accuracy
print("Confusion Matrix:")
print(conf_matrix)
print(f'Accuracy: {accuracy}')


##############################################################

# # single value data test
# data_test = data.iloc[0,:-1].values.reshape(1, -1)

# # Scale the new data point if you scaled your training data
# # new_data_point_scaled = scaler.transform(new_data_point)

# # Making a prediction for the new data point
# prediction = svm_model.predict(data_test)

# print("Predicted Class:", prediction)
