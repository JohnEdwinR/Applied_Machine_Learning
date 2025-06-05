import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

df = pd.read_csv("water_potability.csv")
df.head()

df.isnull().sum()

df.fillna(df.median(), inplace=True)

X = df.drop('Potability', axis=1)
y = df['Potability']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(df.shape)
print(df.info())
df.describe()

sns.countplot(x='Potability', data=df)
plt.title("Potable (1) vs Non-potable (0) Water Samples")
plt.show()

for col in df.columns[:-1]:
  plt.figure(figsize=(6, 4))
  sns.histplot(df[col], kde=True, bins=30)
  plt.title(f'Distribution of {col}')
  plt.show()
  
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

for col in df.columns[:-1]:
  plt.figure(figsize=(6, 4))
  sns.boxplot(x='Potability', y=col, data=df)
  plt.title(f'{col} by Potability')
  plt.show()
  
X = df.drop('Potability', axis=1)
y = df['Potability']
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

from hpelm import ELM
elm = ELM(X_train.shape[1], 1, classification="c")
elm.add_neurons(100, "sigm")
elm.train(X_train, y_train.values.reshape(-1, 1))
y_pred_elm = elm.predict(X_test).flatten().round()
print("ELM Classification Report:")
print(classification_report(y_test, y_pred_elm))

svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm))

mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
print("MLP Classification Report:")
print(classification_report(y_test, y_pred_mlp))
  
xbg = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xbg.fit(X_train, y_train)
y_pred_xgb = xbg.predict(X_test)
print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))

models = {
    'ELM': y_pred_elm,
    'SVM': y_pred_svm,
    'MLP': y_pred_mlp,
    'XGBoost': y_pred_xgb
}

for name, pred in models.items():
  acc = accuracy_score(y_test, pred)
  f1 = f1_score(y_test, pred)
  print(f"{name}: Accuracy = {acc:.4f}, F1 Score = {f1:.4f}")
  
accuracies = [0.6418, 0.6921, 0.6174, 0.6540]
f1_scores = [0.4140, 0.4358, 0.4760, 0.4899]
models = ['ELM', 'SVM', 'MLP', 'XGBoost']

plt.figure(figsize=(10, 5))
plt.bar(models, accuracies, color='skyblue', label='Accuracy')
plt.bar(models, f1_scores, color='salmon', alpha=0.7, label='F1 Score')
plt.title("Model Comparison: Accuracy & F1 Score")
plt.ylabel("Score")
plt.legend()
plt.show()

start = time.time()
elm.train(X_train, y_train.values.reshape(-1, 1))
elm_time = time.time() - start

start = time.time()
svm.fit(X_train, y_train)
svm_time = time.time() - start

start = time.time()
mlp.fit(X_train, y_train)
mlp_time = time.time() - start

start = time.time()
xbg.fit(X_train, y_train)
xgb_time = time.time() - start

print(f"ELM Train Time: {elm_time:.4f}s")
print(f"SVM Train Time: {svm_time:.4f}s")
print(f"MLP Train Time: {mlp_time:.4f}s")
print(f"XGBoost Train Time: {xgb_time:.4f}s")

cm = confusion_matrix(y_test, y_pred_elm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("ELM Confusion Matrix")
plt.show()

X = df.drop('Potability', axis=1)
y = df['Potability']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
unique, counts = np.unique(y_train_smote, return_counts=True)
print(dict(zip(unique, counts)))

elm = ELM(X_train_smote.shape[1], 1, classification="c")
elm.add_neurons(100, "sigm")

elm.train(X_train_smote, y_train_smote.values.reshape(-1, 1))
y_pred_elm = elm.predict(X_test).flatten().round()
svm.fit(X_train_smote, y_train_smote)
y_pred_svm = svm.predict(X_test)
mlp.fit(X_train_smote, y_train_smote)
y_pred_mlp = mlp.predict(X_test)
xbg.fit(X_train_smote, y_train_smote)
y_pred_xgb = xbg.predict(X_test)

models = {
    'ELM': y_pred_elm,
    'SVM': y_pred_svm,
    'MLP': y_pred_mlp,
    'XGBoost': y_pred_xgb
}

for name, pred in models.items():
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    print(f"{name}: Accuracy = {acc:.4f}, F1 Score = {f1:.4f}")
    
joblib.dump(elm, 'elm_water_model.pkl')    
