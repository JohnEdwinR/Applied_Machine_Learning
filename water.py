import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from hpelm import ELM
from imblearn.over_sampling import SMOTE
import time
import shap

class WaterPotabilityAnalysis:
    def __init__(self):
        """Initialize the class with necessary models and parameters"""
        self.scaler = StandardScaler()
        self.elm = None
        self.svm = SVC(probability=True)  # probability=True for SHAP KernelExplainer
        self.mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300)
        self.xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        self.df = None  # DataFrame will be loaded later

    def load_and_preprocess_data(self, filepath):
        """Load and preprocess the dataset"""
        self.df = pd.read_csv(filepath)
        self.df.fillna(self.df.median(), inplace=True)
        X = self.df.drop('Potability', axis=1)
        y = self.df['Potability']
        X_scaled = self.scaler.fit_transform(X)
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    def visualize_data(self):
        """Create visualizations for data analysis"""
        plt.figure(figsize=(8, 6))
        sns.countplot(x='Potability', data=self.df)
        plt.title("Potable (1) vs Non-potable (0) Water Samples")
        plt.show()

        for col in self.df.columns[:-1]:
            plt.figure(figsize=(6, 4))
            sns.histplot(self.df[col], kde=True, bins=30)
            plt.title(f'Distribution of {col}')
            plt.show()

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
        self.elm = ELM(X_train.shape[1], 1, classification="c")
        self.elm.add_neurons(100, "sigm")

        training_times = {}

        start = time.time()
        self.elm.train(X_train, y_train.values.reshape(-1, 1))
        training_times['ELM'] = time.time() - start

        start = time.time()
        self.svm.fit(X_train, y_train)
        training_times['SVM'] = time.time() - start

        start = time.time()
        self.mlp.fit(X_train, y_train)
        training_times['MLP'] = time.time() - start

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

    def explain_models(self, X_train, X_test):
        """Explain models using SHAP"""
        print("Starting SHAP explainability...")

        # XGBoost explainer
        explainer_xgb = shap.Explainer(self.xgb)
        shap_values_xgb = explainer_xgb(X_test)
        print("Showing SHAP summary plot for XGBoost")
        shap.summary_plot(shap_values_xgb, X_test, feature_names=self.df.columns[:-1])

        # Sample train and test data for KernelExplainer
        sample_X_train = shap.sample(X_train, 100, random_state=42)
        sample_X_test = shap.sample(X_test, 50, random_state=42)

        # Convert to numpy arrays explicitly
        sample_X_train_np = np.array(sample_X_train)
        sample_X_test_np = np.array(sample_X_test)

        # MLP KernelExplainer
        try:
            explainer_mlp = shap.KernelExplainer(self.mlp.predict_proba, sample_X_train_np)
            shap_values_mlp = explainer_mlp.shap_values(sample_X_test_np)
            print("Showing SHAP summary plot for MLP")
            
            # Debug: Print shapes to understand the structure
            print(f"MLP SHAP values type: {type(shap_values_mlp)}")
            if isinstance(shap_values_mlp, list):
                print(f"MLP SHAP values length: {len(shap_values_mlp)}")
                print(f"MLP SHAP values[1] shape: {shap_values_mlp[1].shape}")
                print(f"Sample test data shape: {sample_X_test_np.shape}")
                # For binary classification, use the positive class (index 1)
                shap.summary_plot(shap_values_mlp[1], sample_X_test_np, feature_names=self.df.columns[:-1])
            else:
                print(f"MLP SHAP values shape: {shap_values_mlp.shape}")
                # If it's not a list, use it directly
                shap.summary_plot(shap_values_mlp, sample_X_test_np, feature_names=self.df.columns[:-1])
        except Exception as e:
            print(f"Error with MLP SHAP explanation: {e}")
            print("Skipping MLP SHAP plot...")

        # SVM KernelExplainer
        try:
            explainer_svm = shap.KernelExplainer(self.svm.predict_proba, sample_X_train_np)
            shap_values_svm = explainer_svm.shap_values(sample_X_test_np)
            print("Showing SHAP summary plot for SVM")
            
            # Debug: Print shapes to understand the structure
            print(f"SVM SHAP values type: {type(shap_values_svm)}")
            if isinstance(shap_values_svm, list):
                print(f"SVM SHAP values length: {len(shap_values_svm)}")
                print(f"SVM SHAP values[1] shape: {shap_values_svm[1].shape}")
                print(f"Sample test data shape: {sample_X_test_np.shape}")
                # For binary classification, use the positive class (index 1)
                shap.summary_plot(shap_values_svm[1], sample_X_test_np, feature_names=self.df.columns[:-1])
            else:
                print(f"SVM SHAP values shape: {shap_values_svm.shape}")
                # If it's not a list, use it directly
                shap.summary_plot(shap_values_svm, sample_X_test_np, feature_names=self.df.columns[:-1])
        except Exception as e:
            print(f"Error with SVM SHAP explanation: {e}")
            print("Skipping SVM SHAP plot...")

        print("SHAP explainability done.")

def main():
    analysis = WaterPotabilityAnalysis()

    X_train, X_test, y_train, y_test = analysis.load_and_preprocess_data("water_potability.csv")

    analysis.visualize_data()

    X_train_smote, y_train_smote = analysis.apply_smote(X_train, y_train)

    training_times = analysis.train_models(X_train_smote, y_train_smote)

    results = analysis.evaluate_models(X_test, y_test)

    for model, metrics in results.items():
        print(f"\n{model} Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Training Time: {training_times[model]:.4f}s")
        print("Classification Report:")
        print(metrics['report'])

    analysis.plot_results(results)

    analysis.explain_models(X_train_smote, X_test)

if __name__ == "__main__":
    main()