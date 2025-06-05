import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('COVID-19_CBC_Data.csv')

# Quick peek
print(df.head())

# Check missing values
print(df.isnull().sum())

# Prepare target vector (mapping Outcome to 0/1)
y = df['Outcome'].map({'Recovered': 0, 'Not Recovered': 1})

# Drop target and non-numeric date columns before encoding and splitting
X = df.drop(['Outcome', 'Admission_DATE', 'Discharge_DATE or date of Death', 'Sample Collection Date'], axis=1)

# Encode categorical columns
X['Gender'] = X['Gender'].map({'Male': 0, 'Female': 1})
X['Ventilated (Y/N)'] = X['Ventilated (Y/N)'].map({'N': 0, 'Y': 1})
X['What kind of Treatment provided'] = X['What kind of Treatment provided'].astype('category').cat.codes

# Split dataset
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Impute missing numeric values with median
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train_raw)
X_test_imputed = imputer.transform(X_test_raw)

# Scale features with StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Define ELM Classifier
class ELMClassifier:
    def __init__(self, input_size, hidden_size, activation='sigmoid', random_state=42):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation
        np.random.seed(random_state)
        self._initialize_weights()

    def _initialize_weights(self):
        self.W = np.random.randn(self.input_size, self.hidden_size)
        self.b = np.random.randn(self.hidden_size)

    def _activate(self, X):
        H = np.dot(X, self.W) + self.b
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-H))
        elif self.activation == 'relu':
            return np.maximum(0, H)
        elif self.activation == 'tanh':
            return np.tanh(H)
        else:
            raise ValueError("Unsupported activation function.")

    def fit(self, X, y):
        H = self._activate(X)
        lambda_reg = 1e-3
        HTH = H.T @ H
        identity = np.eye(HTH.shape[0])
        self.beta = np.linalg.inv(HTH + lambda_reg * identity) @ H.T @ y

    def predict(self, X):
        H = self._activate(X)
        y_pred = np.dot(H, self.beta)
        return (y_pred > 0.5).astype(int)

# Train ELM
elm = ELMClassifier(input_size=X_train_scaled.shape[1], hidden_size=500, activation='sigmoid')
elm.fit(X_train_scaled, y_train.values)
y_pred_elm = elm.predict(X_test_scaled)

# Train other models
log_reg = LogisticRegression(random_state=42, max_iter=1000)
rf = RandomForestClassifier(random_state=42)
svm = SVC(random_state=42)

log_reg.fit(X_train_scaled, y_train)
rf.fit(X_train_scaled, y_train)
svm.fit(X_train_scaled, y_train)

y_pred_lr = log_reg.predict(X_test_scaled)
y_pred_rf = rf.predict(X_test_scaled)
y_pred_svm = svm.predict(X_test_scaled)

# Metrics printing function
def print_metrics(name, y_true, y_pred):
    print(f"{name} Performance:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
    print("-" * 30)

# Print model metrics
print_metrics("ELM", y_test, y_pred_elm)
print_metrics("Logistic Regression", y_test, y_pred_lr)
print_metrics("Random Forest", y_test, y_pred_rf)
print_metrics("SVM", y_test, y_pred_svm)

# Plot accuracy comparison
models = ['ELM', 'Logistic Regression', 'Random Forest', 'SVM']
accuracies = [
    accuracy_score(y_test, y_pred_elm),
    accuracy_score(y_test, y_pred_lr),
    accuracy_score(y_test, y_pred_rf),
    accuracy_score(y_test, y_pred_svm)
]

plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color='skyblue')
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()
