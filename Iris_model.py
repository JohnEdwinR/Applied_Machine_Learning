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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import hpelm
import time

# Loading the dataset using Scikit-learn
iris = load_iris()

# Displaying the first few rows of the iris dataset
iris_data = sns.load_dataset("iris")
print("First 5 rows of the dataset:")
print(iris_data.head())

# Displaying dataset information
print("\nDisplaying iris data information")
iris_data.info()

# Checking the basic statistic of the dataset
print("\nBasic statistics:")
print(iris_data.describe())

# Identifying categorical and numerical columns
categorical_cols = iris_data.select_dtypes(include=['object']).columns
numerical_cols = iris_data.select_dtypes(include=['int64', 'float64']).columns

# List of categorical features
categorical_features = iris_data.select_dtypes(include=['object']).columns

# Displaying the categorical features
print("\nCategorical Features:")
for feature in categorical_features:
    print(f"- {feature}")

# Display data type of the columns
print("\nData Type of Categorical Features:")
print(iris_data[categorical_features].dtypes)

# Displaying the unique values and their counts relevant to each categorical column
print("\nUnique values and their count relevant to each categorical column:\n")
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
    print("There are no values with 'Unknown' or 'N/A' in the categorical columns")

# Numerical Features
numerical_features = iris_data.select_dtypes(include=['int64', 'float64']).columns

# Displaying the Numerical Columns
print("\nNumerical Features:")
print(numerical_features)

# Displaying the unique values and their count in the numerical columns
print("\nUnique values and their count in the numerical columns:\n")
for col in numerical_features:
    unique_values = iris_data[col].unique()
    value_counts = iris_data[col].value_counts()
    print(f"{col}:")
    print(value_counts.head())  # Show only first few for brevity
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
    plt.subplot(2, 2, i)
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

# VIF Analysis
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Select only numerical columns
X = iris_data.select_dtypes(include=['number'])

# Compute VIF for each numerical feature
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print("\nVIF Analysis:")
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
print("\nOutlier Analysis:")
print(outlier_data)

# Visualizing the Outliers - Box Plot
plt.figure(figsize=(12, 6))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x=iris_data[col])
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)
plt.tight_layout()
plt.show()

# Visualizing the Distribution of the Target Variable
plt.figure(figsize=(8, 6))
sns.countplot(data=iris_data, x='species')
plt.title('Distribution of Species')
plt.xlabel('Species')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Data Preprocessing
# Step 1: Removing whitespaces from the object type columns
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

print(f"\nOriginal data shape: {iris_data.shape}")
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
coef_data = pd.DataFrame(coefficients.reshape(1, -1), columns=X.columns, index=['Coefficient'])
print("\nRidge Regression Coefficients:")
print(coef_data)

# Visualizing the Heatmap after the handling of the multicollinearity
plt.figure(figsize=(10, 4))
sns.heatmap(coef_data, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Importance (Ridge Regression Coefficients)')
plt.show()

# Step 8: Splitting the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize dictionary to store training times and accuracies
model_performance = {}

# Random Forest Model with GridSearchCV
print("\n=== Random Forest Model ===")
rf_start_time = time.time()

rf = RandomForestClassifier(random_state=42)

# Define the parameter grid for grid search
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

# Setting up GridSearchCV with cross-validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)

# Fitting GridSearchCV to the training data
grid_search.fit(X_train, y_train)

rf_end_time = time.time()
rf_training_time = rf_end_time - rf_start_time

# Displaying the best parameters found by GridSearchCV
print("Best Parameters:", grid_search.best_params_)

# Getting the best model from the grid search
best_rf = grid_search.best_estimator_

# Predicting on the test set
y_pred_rf = best_rf.predict(X_test)

# Evaluate the model's performance
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf:.4f}")
print(f"Random Forest Training Time: {rf_training_time:.4f} seconds")

# Store performance metrics
model_performance['Random Forest'] = {
    'accuracy': accuracy_rf,
    'training_time': rf_training_time
}

print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Plotting the Confusion matrix for Random Forest
plt.figure(figsize=(8, 6))
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Saving the model and the scaler for future use
joblib.dump(best_rf, 'iris_rf_model.pkl')
joblib.dump(scaler, 'rf_scaler.pkl')

# SVM Model with Hyperparameter Tuning using GridSearchCV
print("\n=== SVM Model ===")
svm_start_time = time.time()

param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# Instantiating the SVM model
svm_model = SVC()

# Applying GridSearchCV with 5-fold cross-validation
grid_search_svm = GridSearchCV(estimator=svm_model, param_grid=param_grid_svm, cv=5, n_jobs=-1, verbose=1)
grid_search_svm.fit(X_train, y_train)

svm_end_time = time.time()
svm_training_time = svm_end_time - svm_start_time

# Best parameters from GridSearchCV
print("Best SVM Parameters:", grid_search_svm.best_params_)

# Getting the best model
best_svm_model = grid_search_svm.best_estimator_

# Evaluating the best model on the test set
y_pred_svm = best_svm_model.predict(X_test)

# Cross-validation score
cv_scores = cross_val_score(best_svm_model, X_train, y_train, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Average Cross-validation score: {cv_scores.mean():.4f}")

# Print the classification report with the model accuracy and confusion matrix
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {accuracy_svm:.4f}")
print(f"SVM Training Time: {svm_training_time:.4f} seconds")

# Store performance metrics
model_performance['SVM'] = {
    'accuracy': accuracy_svm,
    'training_time': svm_training_time
}

print("\nSVM Classification Report:")
print(classification_report(y_test, y_pred_svm))

# Save the best SVM model
joblib.dump(best_svm_model, 'best_svm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Visualize the SVM confusion matrix
plt.figure(figsize=(8, 6))
cm_svm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('SVM Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# ELM Model
print("\n=== ELM Model ===")
elm_start_time = time.time()

# Convert to numpy arrays if needed
if hasattr(y_train, 'to_numpy'):
    y_train_arr = y_train.to_numpy()
else:
    y_train_arr = y_train

if hasattr(y_test, 'to_numpy'):
    y_test_arr = y_test.to_numpy()
else:
    y_test_arr = y_test

# One-hot encode the target for ELM
encoder = OneHotEncoder(sparse_output=False)
y_train_enc = encoder.fit_transform(y_train_arr.reshape(-1, 1))
y_test_enc = encoder.transform(y_test_arr.reshape(-1, 1))

# Ensure X arrays are numpy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)

# Create and train ELM model
model = hpelm.ELM(X_train.shape[1], y_train_enc.shape[1], classification='c')
model.add_neurons(20, 'sigm')
model.train(X_train, y_train_enc)

elm_end_time = time.time()
elm_training_time = elm_end_time - elm_start_time

# Evaluating the ELM model on the test set
y_pred_elm = model.predict(X_test)
y_pred_labels = np.argmax(y_pred_elm, axis=1)
y_true_labels = np.argmax(y_test_enc, axis=1)

# Accuracy of the ELM Model
accuracy_elm = accuracy_score(y_true_labels, y_pred_labels)
print(f"ELM Test Accuracy: {accuracy_elm:.4f}")
print(f"ELM Training Time: {elm_training_time:.4f} seconds")

# Store performance metrics
model_performance['ELM'] = {
    'accuracy': accuracy_elm,
    'training_time': elm_training_time
}

# Visualization of the ELM Confusion Matrix (Fixed)
cm_elm = confusion_matrix(y_true_labels, y_pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_elm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('ELM Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Model Comparison
print("\n=== Comprehensive Model Comparison ===")

# Create comparison DataFrame
models_comparison = pd.DataFrame({
    'Model': list(model_performance.keys()),
    'Accuracy': [model_performance[model]['accuracy'] for model in model_performance.keys()],
    'Training Time (seconds)': [model_performance[model]['training_time'] for model in model_performance.keys()]
})

# Add efficiency metric (accuracy per second)
models_comparison['Efficiency (Accuracy/Time)'] = models_comparison['Accuracy'] / models_comparison['Training Time (seconds)']

# Sort by accuracy (descending)
models_comparison_sorted = models_comparison.sort_values('Accuracy', ascending=False)

print("Model Performance Comparison:")
print("=" * 80)
print(models_comparison_sorted.to_string(index=False, float_format='%.4f'))

# Find best performing models
best_accuracy_model = models_comparison_sorted.iloc[0]['Model']
fastest_model = models_comparison.loc[models_comparison['Training Time (seconds)'].idxmin(), 'Model']
most_efficient_model = models_comparison.loc[models_comparison['Efficiency (Accuracy/Time)'].idxmax(), 'Model']

print(f"\n=== Performance Summary ===")
print(f"🏆 Best Accuracy: {best_accuracy_model} ({models_comparison_sorted.iloc[0]['Accuracy']:.4f})")
print(f"⚡ Fastest Training: {fastest_model} ({models_comparison.loc[models_comparison['Training Time (seconds)'].idxmin(), 'Training Time (seconds)']:.4f} seconds)")
print(f"⚖️  Most Efficient: {most_efficient_model} (Efficiency: {models_comparison['Efficiency (Accuracy/Time)'].max():.4f})")

# Visualization of model comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Accuracy comparison
axes[0, 0].bar(models_comparison['Model'], models_comparison['Accuracy'], color=['skyblue', 'lightcoral', 'lightgreen'])
axes[0, 0].set_title('Model Accuracy Comparison')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_ylim(0, 1)
for i, v in enumerate(models_comparison['Accuracy']):
    axes[0, 0].text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')

# Training time comparison
axes[0, 1].bar(models_comparison['Model'], models_comparison['Training Time (seconds)'], color=['skyblue', 'lightcoral', 'lightgreen'])
axes[0, 1].set_title('Model Training Time Comparison')
axes[0, 1].set_ylabel('Training Time (seconds)')
for i, v in enumerate(models_comparison['Training Time (seconds)']):
    axes[0, 1].text(i, v + 0.1, f'{v:.2f}s', ha='center', fontweight='bold')

# Efficiency comparison
axes[1, 0].bar(models_comparison['Model'], models_comparison['Efficiency (Accuracy/Time)'], color=['skyblue', 'lightcoral', 'lightgreen'])
axes[1, 0].set_title('Model Efficiency Comparison (Accuracy/Time)')
axes[1, 0].set_ylabel('Efficiency (Accuracy per Second)')
for i, v in enumerate(models_comparison['Efficiency (Accuracy/Time)']):
    axes[1, 0].text(i, v + 0.001, f'{v:.3f}', ha='center', fontweight='bold')

# Scatter plot: Accuracy vs Training Time
axes[1, 1].scatter(models_comparison['Training Time (seconds)'], models_comparison['Accuracy'], 
                   s=200, c=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.7, edgecolors='black')
axes[1, 1].set_xlabel('Training Time (seconds)')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].set_title('Accuracy vs Training Time Trade-off')

# Add model labels to scatter plot
for i, model in enumerate(models_comparison['Model']):
    axes[1, 1].annotate(model, 
                       (models_comparison['Training Time (seconds)'].iloc[i], 
                        models_comparison['Accuracy'].iloc[i]),
                       xytext=(5, 5), textcoords='offset points', fontweight='bold')

plt.tight_layout()
plt.show()

# Create a detailed performance table
print("\n=== Detailed Performance Analysis ===")
performance_analysis = models_comparison.copy()
performance_analysis['Accuracy Rank'] = performance_analysis['Accuracy'].rank(ascending=False).astype(int)
performance_analysis['Speed Rank'] = performance_analysis['Training Time (seconds)'].rank(ascending=True).astype(int)
performance_analysis['Efficiency Rank'] = performance_analysis['Efficiency (Accuracy/Time)'].rank(ascending=False).astype(int)
performance_analysis['Overall Rank'] = (performance_analysis['Accuracy Rank'] + 
                                      performance_analysis['Speed Rank'] + 
                                      performance_analysis['Efficiency Rank']) / 3

print(performance_analysis.to_string(index=False, float_format='%.4f'))

# Function to make predictions with sample data instead of user input
def make_sample_predictions():
    """Make predictions using sample data instead of interactive input"""
    
    # Sample flower measurements (you can change these values)
    sample_features = [
        [5.1, 3.5, 1.4, 0.2],  # Likely Setosa
        [6.2, 2.9, 4.3, 1.3],  # Likely Versicolor  
        [7.3, 2.9, 6.3, 1.8]   # Likely Virginica
    ]
    
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    
    print("=== Sample Predictions ===")
    
    for i, features in enumerate(sample_features):
        print(f"\nSample {i+1}: {dict(zip(feature_names, features))}")
        
        # Scale the features
        user_input_scaled = scaler.transform([features])
        
        # Random Forest prediction
        rf_pred = best_rf.predict(user_input_scaled)
        rf_class = label_encoder.inverse_transform(rf_pred)[0]
        
        # SVM prediction
        svm_pred = best_svm_model.predict(user_input_scaled)
        svm_class = label_encoder.inverse_transform(svm_pred)[0]
        
        # ELM prediction
        elm_pred_probs = model.predict(user_input_scaled)
        elm_pred_class = np.argmax(elm_pred_probs)
        elm_class = iris.target_names[elm_pred_class]
        
        print(f"Random Forest prediction: {rf_class}")
        print(f"SVM prediction: {svm_class}")
        print(f"ELM prediction: {elm_class}")

# Make sample predictions
make_sample_predictions()

# LIME Explanation for sample data
print("\n=== LIME Explanation (for Ridge model) ===")

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
explanation = explainer.explain_instance(instance[0], ridge.predict, num_features=4)

# Display LIME explanation text
print("LIME Explanation for Ridge model:")
print("Instance features:", dict(zip(X.columns, instance[0])))
print("Prediction:", ridge.predict(instance)[0])

# Print the explanation
for feature, importance in explanation.as_list():
    print(f"{feature}: {importance:.4f}")

print("\nAnalysis complete! All models have been trained and evaluated.")
print("Saved models: iris_rf_model.pkl, best_svm_model.pkl, label_encoder.pkl, scaler.pkl")