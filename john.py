import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="🌸 Iris Classification Dashboard",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .prediction-result {
        font-size: 1.5rem;
        font-weight: bold;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .setosa { background-color: #ff9999; }
    .versicolor { background-color: #66b3ff; }
    .virginica { background-color: #99ff99; }
</style>
""", unsafe_allow_html=True)

# Load models and preprocessors (with error handling)
@st.cache_resource
def load_models():
    try:
        models = {}
        models['rf'] = joblib.load('iris_rf_model.pkl')
        models['svm'] = joblib.load('best_svm_model.pkl')
        models['scaler'] = joblib.load('scaler.pkl')
        models['label_encoder'] = joblib.load('label_encoder.pkl')
        return models
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.info("Please ensure all model files are in the same directory as this app.")
        return None

# Load sample data
@st.cache_data
def load_sample_data():
    # Using seaborn's iris dataset
    iris_data = sns.load_dataset('iris')
    return iris_data

def main():
    # Header
    st.markdown('<h1 class="main-header">🌸 Iris Classification Dashboard</h1>', unsafe_allow_html=True)
    
    # Load models
    models = load_models()
    if models is None:
        st.stop()
    
    # Load sample data
    iris_data = load_sample_data()
    
    # Sidebar
    st.sidebar.header("🎛️ Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home & Prediction", "📊 Data Exploration", "📈 Model Performance", "🔍 Feature Analysis"]
    )
    
    if page == "🏠 Home & Prediction":
        show_prediction_page(models, iris_data)
    elif page == "📊 Data Exploration":
        show_data_exploration(iris_data)
    elif page == "📈 Model Performance":
        show_model_performance(models, iris_data)
    elif page == "🔍 Feature Analysis":
        show_feature_analysis(iris_data)

def show_prediction_page(models, iris_data):
    st.header("🔮 Make Predictions")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🌸 Enter Flower Measurements")
        
        # Input sliders with realistic ranges
        sepal_length = st.slider(
            "Sepal Length (cm)", 
            min_value=4.0, max_value=8.0, value=5.8, step=0.1,
            help="Length of the sepal in centimeters"
        )
        
        sepal_width = st.slider(
            "Sepal Width (cm)", 
            min_value=2.0, max_value=4.5, value=3.0, step=0.1,
            help="Width of the sepal in centimeters"
        )
        
        petal_length = st.slider(
            "Petal Length (cm)", 
            min_value=1.0, max_value=7.0, value=3.8, step=0.1,
            help="Length of the petal in centimeters"
        )
        
        petal_width = st.slider(
            "Petal Width (cm)", 
            min_value=0.1, max_value=2.5, value=1.2, step=0.1,
            help="Width of the petal in centimeters"
        )
        
        # Prediction button
        if st.button("🔍 Predict Species", type="primary"):
            make_predictions(models, [sepal_length, sepal_width, petal_length, petal_width])
    
    with col2:
        st.subheader("📋 Quick Examples")
        
        examples = {
            "Setosa Example": [5.1, 3.5, 1.4, 0.2],
            "Versicolor Example": [6.2, 2.9, 4.3, 1.3],
            "Virginica Example": [7.3, 2.9, 6.3, 1.8]
        }
        
        for name, values in examples.items():
            if st.button(f"Use {name}"):
                st.session_state.example_values = values
                make_predictions(models, values)
        
        # Show iris species information
        st.subheader("🌺 Iris Species Information")
        species_info = {
            "Setosa": "Small flowers with short, wide petals",
            "Versicolor": "Medium-sized flowers with moderate petal dimensions",
            "Virginica": "Large flowers with long, wide petals"
        }
        
        for species, description in species_info.items():
            st.info(f"**{species}**: {description}")

def make_predictions(models, features):
    """Make predictions using all available models"""
    
    # Scale features
    features_scaled = models['scaler'].transform([features])
    
    st.subheader("🎯 Prediction Results")
    
    # Get predictions from both models
    rf_pred = models['rf'].predict(features_scaled)[0]
    svm_pred = models['svm'].predict(features_scaled)[0]
    
    # Get prediction probabilities
    rf_proba = models['rf'].predict_proba(features_scaled)[0]
    svm_proba = models['svm'].predict_proba(features_scaled)[0] if hasattr(models['svm'], 'predict_proba') else None
    
    # Convert predictions to species names
    species_names = ['setosa', 'versicolor', 'virginica']
    rf_species = species_names[rf_pred]
    svm_species = species_names[svm_pred]
    
    # Display predictions
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="prediction-result {rf_species}">
            Random Forest: {rf_species.title()}
        </div>
        """, unsafe_allow_html=True)
        
        # Show probabilities
        if rf_proba is not None:
            prob_df = pd.DataFrame({
                'Species': species_names,
                'Probability': rf_proba
            })
            fig = px.bar(prob_df, x='Species', y='Probability', 
                        title='Random Forest Confidence')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown(f"""
        <div class="prediction-result {svm_species}">
            SVM: {svm_species.title()}
        </div>
        """, unsafe_allow_html=True)
        
        # Show probabilities if available
        if svm_proba is not None:
            prob_df = pd.DataFrame({
                'Species': species_names,
                'Probability': svm_proba
            })
            fig = px.bar(prob_df, x='Species', y='Probability', 
                        title='SVM Confidence')
            st.plotly_chart(fig, use_container_width=True)
    
    # Agreement indicator
    if rf_species == svm_species:
        st.success(f"✅ Both models agree: **{rf_species.title()}**")
    else:
        st.warning(f"⚠️ Models disagree: RF says {rf_species.title()}, SVM says {svm_species.title()}")

def show_data_exploration(iris_data):
    st.header("📊 Data Exploration")
    
    # Dataset overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", len(iris_data))
    with col2:
        st.metric("Features", len(iris_data.columns) - 1)
    with col3:
        st.metric("Species", iris_data['species'].nunique())
    with col4:
        st.metric("Missing Values", iris_data.isnull().sum().sum())
    
    # Display raw data
    st.subheader("📋 Dataset Sample")
    st.dataframe(iris_data.head(10))
    
    # Basic statistics
    st.subheader("📈 Statistical Summary")
    st.dataframe(iris_data.describe())
    
    # Visualizations
    st.subheader("📊 Data Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["Distribution Plots", "Correlation Matrix", "Species Comparison"])
    
    with tab1:
        # Distribution plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
        )
        
        features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        positions = [(1,1), (1,2), (2,1), (2,2)]
        
        for feature, (row, col) in zip(features, positions):
            fig.add_trace(
                go.Histogram(x=iris_data[feature], name=feature, showlegend=False),
                row=row, col=col
            )
        
        fig.update_layout(height=600, title_text="Feature Distributions")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Correlation matrix
        corr_matrix = iris_data.select_dtypes(include=[np.number]).corr()
        
        fig = px.imshow(corr_matrix, 
                       text_auto=True, 
                       aspect="auto",
                       title="Feature Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Species comparison
        feature_to_plot = st.selectbox("Select feature for comparison:", 
                                     ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        
        fig = px.box(iris_data, x='species', y=feature_to_plot, 
                    title=f'{feature_to_plot.replace("_", " ").title()} by Species')
        st.plotly_chart(fig, use_container_width=True)

def show_model_performance(models, iris_data):
    st.header("📈 Model Performance Analysis")
    
    # Simulated performance metrics (replace with actual metrics from your training)
    performance_data = {
        'Model': ['Random Forest', 'SVM', 'ELM'],
        'Accuracy': [0.9667, 0.9667, 0.9333],
        'Training Time (s)': [0.25, 0.15, 0.08],
        'Precision': [0.97, 0.97, 0.93],
        'Recall': [0.97, 0.97, 0.93],
        'F1-Score': [0.97, 0.97, 0.93]
    }
    
    performance_df = pd.DataFrame(performance_data)
    
    # Display metrics table
    st.subheader("📊 Performance Metrics")
    st.dataframe(performance_df)
    
    # Performance visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy comparison
        fig = px.bar(performance_df, x='Model', y='Accuracy', 
                    title='Model Accuracy Comparison',
                    color='Accuracy',
                    color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Training time comparison
        fig = px.bar(performance_df, x='Model', y='Training Time (s)',
                    title='Training Time Comparison',
                    color='Training Time (s)',
                    color_continuous_scale='plasma')
        st.plotly_chart(fig, use_container_width=True)
    
    # Radar chart for comprehensive comparison
    st.subheader("🎯 Comprehensive Model Comparison")
    
    fig = go.Figure()
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    for i, model in enumerate(performance_df['Model']):
        values = [performance_df.iloc[i][metric] for metric in metrics]
        values.append(values[0])  # Close the radar chart
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],
            fill='toself',
            name=model
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Model Performance Radar Chart"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_feature_analysis(iris_data):
    st.header("🔍 Feature Analysis")
    
    # Feature importance (simulated for demonstration)
    st.subheader("🎯 Feature Importance")
    
    importance_data = {
        'Feature': ['Petal Length', 'Petal Width', 'Sepal Length', 'Sepal Width'],
        'Importance': [0.45, 0.35, 0.15, 0.05],
        'Rank': [1, 2, 3, 4]
    }
    
    importance_df = pd.DataFrame(importance_data)
    
    fig = px.bar(importance_df, x='Feature', y='Importance',
                title='Feature Importance (Random Forest)',
                color='Importance',
                color_continuous_scale='viridis')
    st.plotly_chart(fig, use_container_width=True)
    
    # Pairwise relationships
    st.subheader("🔗 Feature Relationships")
    
    fig = px.scatter_matrix(iris_data,
                           dimensions=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
                           color='species',
                           title="Pairwise Feature Relationships")
    st.plotly_chart(fig, use_container_width=True)
    
    # 3D scatter plot
    st.subheader("🌐 3D Feature Visualization")
    
    feature_x = st.selectbox("X-axis:", ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], index=0)
    feature_y = st.selectbox("Y-axis:", ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], index=2)
    feature_z = st.selectbox("Z-axis:", ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], index=3)
    
    fig = px.scatter_3d(iris_data, x=feature_x, y=feature_y, z=feature_z,
                       color='species',
                       title=f'3D Scatter: {feature_x} vs {feature_y} vs {feature_z}')
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()