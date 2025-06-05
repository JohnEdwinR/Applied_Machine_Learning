import numpy as np
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import lime.lime_tabular
from sklearn.svm import SVC
from scipy.stats import zscore
from sklearn.datasets import load_iris, load_svmlight_file
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import hpelm

# Loading the dataset using Scikit-learn
iris = load_iris()

# Displaying the first few rows of the iris dataset
iris_data = sns.load_dataset("iris")
iris_data.head()

# Displaying dataset information
print("Displaying iris data information")
iris_data.info()

# Checking the basic statistic of the dataset
iris_data.describe()

# Identifying categorical and numerical columns
categorical_cols = iris_data.select_dtypes(include=['object']).columns
numerical_cols = iris_data.select_dtypes(include=['int64', 'float64']).columns

# List of categorical features
categorical_features = iris_data.select_dtypes(include=['object']).columns

# Displaying the categorical features
print("Categorical Features:")
for feature in categorical_features:
    print(f"- {feature}")

# Display data type of the columns
print("\nData Type of Categorical Features:")
print(iris_data[categorical_features].dtypes)

# Displaying the unique values and their counts relevant to each categorical column
print("Unique values and their count relevant to each categorical column:\n")
for col in categorical_features:
    unique_values = iris_data[col].unique()
    value_counts = iris_data[col].value_counts()
    print(value_counts)
    print(" ")

# Displaying the categorical columns which contain null values and their counts
found_nulls = False
for col in categorical_features:
    null_count = iris_data[col].isnull().sum()
    if null_count > 0:
        print(f"{col}: {null_count}")
        found_nulls = True

if not found_nulls:
    print("There are no null values in the categorical columns")

# Displaying the categorical columns which contain 'Unknown' or 'N/A' values and their relevant counts
found_unknown_na = False

for col in categorical_features:
    unknown_count = (iris_data[col] == 'Unknown').sum()
    na_count = (iris_data[col] == 'N/A').sum()

    if unknown_count > 0 or na_count > 0:
        found_unknown_na = True
        if unknown_count > 0:
            print(f"{col} - 'Unknown': {unknown_count}")
        if na_count > 0:
            print(f"{col} - 'N/A': {na_count}")

if not found_unknown_na:
    print("There are no values with 'Unknown' or 'N/A' in the catgorical columns")

# Numerical Features
numerical_features = iris_data.select_dtypes(include=['int64', 'float64']).columns

# Displaying the Numerical Columns
print("Numerical Features:")
print(numerical_features)

# Displaying the unique values and their count in the numerical columns
print("Unique values and their count in the numerical columns:\n")
for col in numerical_features:
    unique_values = iris_data[col].unique()
    value_counts = iris_data[col].value_counts()
    print(value_counts)
    print(" ")

# Displaying the numerical columns with null values and their relevant counts
found_nulls = False

for col in numerical_cols:
    null_count = iris_data[col].isnull().sum()
    if null_count > 0:
        print(f"{col}: {null_count}")
        found_nulls = True

if not found_nulls:
    print("There are no null values in the Numerical Columns.")

# Visualizing the distribution of the Numerical Columns
plt.figure(figsize=(12, 6))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 3, i)
    sns.histplot(iris_data[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Visualizing the Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(iris_data[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Select only numerical columns
X = iris_data.select_dtypes(include=['number'])

# Compute VIF for each numerical feature
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif_data)

# Visualizing the Outliers in numerical values

# Selecting only numerical columns
numerical_features = iris_data.select_dtypes(include=['number']).columns

# Creating a DataFrame to store the results
outlier_bounds = []

for col in numerical_features:
    Q1 = iris_data[col].quantile(0.25)
    Q3 = iris_data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier_bounds.append([col, Q1, Q3, IQR, lower_bound, upper_bound])

# Convert to DataFrame for better readability
outlier_data = pd.DataFrame(outlier_bounds,
                          columns=['Feature', 'Q1', 'Q3', 'IQR', 'Lower Bound', 'Upper Bound'])

# Display the DataFrame
print(outlier_data)

# Visualizing the Outliers - Box Plot
plt.figure(figsize=(12, 6))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x=iris_data[col])
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)
plt.tight_layout()
plt.show()

# Visualizing the Distribution of the Target Variable
plt.figure(figsize=(8, 6))
sns.histplot(iris_data['species'], kde=True)
plt.title('Distribution of Species')
plt.xlabel('Species')
plt.ylabel('Frequency')
plt.show()

# # Step 1: Removing whitespaces from the object type columns
object_columns = iris_data.select_dtypes(include=['object']).columns
iris_data[object_columns] = iris_data[object_columns].apply(lambda x: x.str.strip())

# Step 2: Dropping duplicate values
iris_data.drop_duplicates(inplace=True)
iris_data.reset_index(drop=True, inplace=True)

# Display dataset information
print("\nDataset information after removing duplicates:")
iris_data.info()

# Step 3: Encoding the Categorical Columns using the label encoder
label_encoder = LabelEncoder()
iris_data['species'] = label_encoder.fit_transform(iris_data['species'])

# Saving the encoder
joblib.dump(label_encoder, 'label_encoder.pkl')

# Step 4: Handling the outliers in the numerical columns

# Removing outliers from numerical features
filtered_data = iris_data.copy()

for col in numerical_features:
    Q1 = iris_data[col].quantile(0.25)
    Q3 = iris_data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Keeping only values within the bounds
    filtered_data = filtered_data[(filtered_data[col] >= lower_bound) & (filtered_data[col] <= upper_bound)]

print(f"Original data shape: {iris_data.shape}")
print(f"Filtered data shape: {filtered_data.shape}")

# Step 5: Features (X) and Target (y)
X = filtered_data.drop(['species'], axis=1)
y = filtered_data['species']

# Step 6: Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 7: Handling the collinearity in the dataset

# Apply Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_scaled, y)

# Get the coefficients
coefficients = ridge.coef_

# Display the coefficients
coef_data = pd.DataFrame(coefficients, index=X.columns, columns=['Coefficient'])
print(coef_data)

# Visualizing the Heatmap after the handling of the multicollinearity
plt.figure(figsize=(8, 5))
sns.heatmap(coef_data.T, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Importance (Ridge Regression Coefficients)')
plt.xlabel('Coefficient Value')
plt.ylabel('Features')
plt.show()

# Step 8: Splitting the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Defining the RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Define the parameter grid for grid search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Setting up GridSearchCV with cross-validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fitting GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# Displaying the best parameters found by GridSearchCV
print("Best Parameters:", grid_search.best_params_)

# Getting the best model from the grid search
best_rf = grid_search.best_estimator_

# Predicting on the test set
y_pred = best_rf.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plotting the Confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted']),
            annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.show()

# Saving the model and the scaler for future use
joblib.dump(best_rf, 'iris_rf_model.pkl')
joblib.dump(scaler, 'rf_scaler.pkl')

# Creating a LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_scaled,
    feature_names=X.columns,
    class_names=['Target'],
    mode='regression'
)

# Selecting a random instance
instance_index = 10
instance = X_scaled[instance_index, :].reshape(1, -1)

# Explaining the prediction for this instance
explanation = explainer.explain_instance(instance[0], ridge.predict, num_features=5)

# Displaying the explanation
explanation.show_in_notebook()

# Creating a SHAP explainer for Ridge Regression
explainer = shap.Explainer(ridge, X_scaled)

# Selecting an instance (using NumPy indexing)
instance_index = 10
instance = X_scaled[instance_index, :].reshape(1, -1)

# Getting SHAP values for this instance
shap_values = explainer(instance)

# Plotting SHAP values for this specific instance
shap.initjs()
shap.force_plot(shap_values.base_values[0], shap_values.values[0], instance)

# Loading the saved model and encoder
model = joblib.load('iris_rf_model.pkl')
encoder = joblib.load('label_encoder.pkl')

# Getting the prediction

# Getting user inputs
def get_user_input():
    sepal_length = float(input("Enter sepal length: "))
    sepal_width = float(input("Enter sepal width: "))
    petal_length = float(input("Enter petal length: "))
    petal_width = float(input("Enter petal width: "))

    # Returning as a numpy array
    return np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)

# Preprocessing the user input
def preprocess_input(user_input):
    # Scaling the input using the same scaler that was used during training
    scaler = joblib.load('rf_scaler.pkl')
    user_input_scaled = scaler.transform(user_input)

    return user_input_scaled

# Getting the user input
user_input = get_user_input()

# Preprocessing the input
user_input_scaled = preprocess_input(user_input)

# Making prediction using the trained model
prediction = model.predict(user_input_scaled)

# Decoding the prediction
predicted_class = encoder.inverse_transform(prediction)

# Displaying the result
print(f"\nPredicted class: {predicted_class[0]}")

# SVM Model with Hyperparameter Tuning using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto'],
    'degree': [3, 5]
}

# Instantiating the SVM model
svm_model = SVC()

# Applying GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Best parameters from GridSearchCV
print("Best Parameters from GridSearchCV:", grid_search.best_params_)

# Getting the best model
best_svm_model = grid_search.best_estimator_

# Evaluating the best model on the test set
y_pred = best_svm_model.predict(X_test)

# Cross-validation score (to evaluate model performance more robustly)
cv_scores = cross_val_score(best_svm_model, X_train, y_train, cv=5)
print(f"\nCross-validation scores: {cv_scores}")
print(f"Average Cross-validation score: {cv_scores.mean()}")

# Print the classification report with the model accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Save the best model and scaler for later use
joblib.dump(best_svm_model, 'best_svm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Confusion Matrix for SVM Classifier')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Creating a LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    feature_names=X.columns,
    class_names=['Target'],
    mode='regression'
)

# Selecting a random instance
instance_index = 10
instance = X_train[instance_index, :].reshape(1, -1)

# Explaining the prediction for this instance
explanation = explainer.explain_instance(instance[0], ridge.predict, num_features=5)

# Displaying the explanation
explanation.show_in_notebook()

# Creating a SHAP explainer for Ridge Regression
explainer = shap.Explainer(ridge, X_train)

# Selecting an instance (using NumPy indexing)
instance_index = 10
instance = X_train[instance_index, :].reshape(1, -1)

# Getting SHAP values for this instance
shap_values = explainer(instance)

# Plotting SHAP values for this specific instance
shap.initjs()
shap.force_plot(shap_values.base_values[0], shap_values.values[0], instance)

# Loading the saved model and encoder
svm_model = joblib.load('best_svm_model.pkl')
encoder = joblib.load('label_encoder.pkl')

# Getting the prediction

# Getting user inputs
def get_user_input():
    sepal_length = float(input("Enter sepal length: "))
    sepal_width = float(input("Enter sepal width: "))
    petal_length = float(input("Enter petal length: "))
    petal_width = float(input("Enter petal width: "))

    # Returning as a numpy array
    return np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)

# Preprocessing the user input
def preprocess_input(user_input):
    # Scaling the input using the same scaler that was used during training
    scaler = joblib.load('scaler.pkl')
    user_input_scaled = scaler.transform(user_input)

    return user_input_scaled

# Getting the user input
user_input = get_user_input()

# Preprocessing the input
user_input_scaled = preprocess_input(user_input)

# Making prediction using the trained model
prediction = svm_model.predict(user_input_scaled)

# Decoding the prediction
predicted_class = encoder.inverse_transform(prediction)

# Displaying the result
print(f"\nPredicted class: {predicted_class[0]}")

# ELM Grid Search for Best Model
hidden_neurons = [10, 20, 50]
activations = ['sigm', 'tanh', 'rbf_l1']

best_acc = 0
best_config = None
best_model = None

if hasattr(y_train, 'to_numpy'):
    y_train_arr = y_train.to_numpy()
else:
    y_train_arr = y_train

if hasattr(y_test, 'to_numpy'):
    y_test_arr = y_test.to_numpy()
else:
    y_test_arr = y_test

encoder = OneHotEncoder(sparse_output=False)
y_train_enc = encoder.fit_transform(y_train_arr.reshape(-1, 1))
y_test_enc = encoder.transform(y_test_arr.reshape(-1, 1))

X_train = np.array(X_train)
X_test = np.array(X_test)

model = hpelm.ELM(X_train.shape[1], y_train_enc.shape[1], classification='c')
model.add_neurons(20, 'sigm')
model.train(X_train, y_train_enc)

# Evaluating the best model on the test set
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test_enc, axis=1)

# Accuracy of the ELM Model
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true_labels, y_pred_labels)

print(f"Test Accuracy: {accuracy:.4f}")

# Visualization of the Confusion Matrix

cm = confusion_matrix(y_true_labels, y_pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.categories_[0])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix on Test Set')
plt.show()

# Getting the prediction
print("\nEnter flower features for prediction:")
features = []
for name in iris.feature_names:
    val = float(input(f"{name}: "))
    features.append(val)

user_input = scaler.transform([features])
pred_probs = model.predict(user_input)
pred_class = np.argmax(pred_probs)

print(f"\nPredicted Class: {iris.target_names[pred_class]}")

# LIME Explanation
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def elm_predict_proba(x):
    return softmax(model.predict(x))

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    mode='classification'
)

lime_exp = lime_explainer.explain_instance(user_input[0], elm_predict_proba, num_features=4)
lime_exp.show_in_notebook(show_table=True)