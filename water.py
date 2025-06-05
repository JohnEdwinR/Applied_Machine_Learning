import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from hpelm import ELM
from imblearn.over_sampling import SMOTE
import time

class WaterPotabilityAnalysis:
    def __init__(self):
        """Initialize the class with necessary models and parameters"""
        self.scaler = StandardScaler()
        self.elm = None
        self.svm = SVC()
        self.mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300)
        self.xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        
    def load_and_preprocess_data(self, filepath):
        """Load and preprocess the dataset"""
        # Load data
        self.df = pd.read_csv(filepath)
        
        # Handle missing values
        self.df.fillna(self.df.median(), inplace=True)
        
        # Split features and target
        X = self.df.drop('Potability', axis=1)
        y = self.df['Potability']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    def visualize_data(self):
        """Create visualizations for data analysis"""
        # Distribution of potable vs non-potable water
        plt.figure(figsize=(8, 6))
        sns.countplot(x='Potability', data=self.df)
        plt.title("Potable (1) vs Non-potable (0) Water Samples")
        plt.show()
        
        # Feature distributions
        for col in self.df.columns[:-1]:
            plt.figure(figsize=(6, 4))
            sns.histplot(self.df[col], kde=True, bins=30)
            plt.title(f'Distribution of {col}')
            plt.show()
        
        # Correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Matrix")
        plt.show()
    
    def apply_smote(self, X_train, y_train):
        """Apply SMOTE for handling class imbalance"""
        smote = SMOTE(random_state=42)
        return smote.fit_resample(X_train, y_train)
    
    def train_models(self, X_train, y_train):
        """Train all models and measure training time"""
        # Initialize ELM
        self.elm = ELM(X_train.shape[1], 1, classification="c")
        self.elm.add_neurons(100, "sigm")
        
        # Dictionary to store training times
        training_times = {}
        
        # Train ELM
        start = time.time()
        self.elm.train(X_train, y_train.values.reshape(-1, 1))
        training_times['ELM'] = time.time() - start
        
        # Train SVM
        start = time.time()
        self.svm.fit(X_train, y_train)
        training_times['SVM'] = time.time() - start
        
        # Train MLP
        start = time.time()
        self.mlp.fit(X_train, y_train)
        training_times['MLP'] = time.time() - start
        
        # Train XGBoost
        start = time.time()
        self.xgb.fit(X_train, y_train)
        training_times['XGBoost'] = time.time() - start
        
        return training_times
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all models and return performance metrics"""
        predictions = {
            'ELM': self.elm.predict(X_test).flatten().round(),
            'SVM': self.svm.predict(X_test),
            'MLP': self.mlp.predict(X_test),
            'XGBoost': self.xgb.predict(X_test)
        }
        
        results = {}
        for name, pred in predictions.items():
            results[name] = {
                'accuracy': accuracy_score(y_test, pred),
                'f1': f1_score(y_test, pred),
                'report': classification_report(y_test, pred)
            }
        
        return results
    
    def plot_results(self, results):
        """Plot comparison of model performances"""
        accuracies = [results[model]['accuracy'] for model in results]
        f1_scores = [results[model]['f1'] for model in results]
        models = list(results.keys())
        
        plt.figure(figsize=(10, 5))
        plt.bar(models, accuracies, color='skyblue', label='Accuracy')
        plt.bar(models, f1_scores, color='salmon', alpha=0.7, label='F1 Score')
        plt.title("Model Comparison: Accuracy & F1 Score")
        plt.ylabel("Score")
        plt.legend()
        plt.show()

def main():
    # Initialize analysis
    analysis = WaterPotabilityAnalysis()
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = analysis.load_and_preprocess_data("water_potability.csv")
    
    # Visualize data
    analysis.visualize_data()
    
    # Apply SMOTE to handle class imbalance
    X_train_smote, y_train_smote = analysis.apply_smote(X_train, y_train)
    
    # Train models and get training times
    training_times = analysis.train_models(X_train_smote, y_train_smote)
    
    # Evaluate models
    results = analysis.evaluate_models(X_test, y_test)
    
    # Display results
    for model, metrics in results.items():
        print(f"\n{model} Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Training Time: {training_times[model]:.4f}s")
        print("\nClassification Report:")
        print(metrics['report'])
    
    # Plot results
    analysis.plot_results(results)

if __name__ == "__main__":
    main()