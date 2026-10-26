import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
from typing import Dict, Any, List, Tuple
import streamlit as st

def plot_training_history(history: Dict[str, List[float]]) -> go.Figure:
    """
    Plot training history using Plotly
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Model Loss', 'Model Accuracy'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    epochs = list(range(1, len(history['loss']) + 1))
    
    # Loss plot
    fig.add_trace(
        go.Scatter(x=epochs, y=history['loss'], name='Training Loss', line=dict(color='blue')),
        row=1, col=1
    )
    
    if 'val_loss' in history and history['val_loss']:
        fig.add_trace(
            go.Scatter(x=epochs, y=history['val_loss'], name='Validation Loss', line=dict(color='red')),
            row=1, col=1
        )
    
    # Accuracy plot
    fig.add_trace(
        go.Scatter(x=epochs, y=history['accuracy'], name='Training Accuracy', line=dict(color='blue')),
        row=1, col=2
    )
    
    if 'val_accuracy' in history and history['val_accuracy']:
        fig.add_trace(
            go.Scatter(x=epochs, y=history['val_accuracy'], name='Validation Accuracy', line=dict(color='red')),
            row=1, col=2
        )
    
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)
    
    fig.update_layout(height=400, showlegend=True, title_text="Training Progress")
    
    return fig

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str] = None) -> go.Figure:
    """
    Plot confusion matrix using Plotly
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongap=0
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        height=400
    )
    
    return fig

def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                   class_names: List[str] = None) -> Dict[str, Any]:
    """
    Calculate comprehensive classification metrics
    """
    # Convert predictions to class labels if they're probabilities
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred_classes = np.argmax(y_pred, axis=1)
    else:
        y_pred_classes = y_pred.astype(int)
    
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true_classes = np.argmax(y_true, axis=1)
    else:
        y_true_classes = y_true.astype(int)
    
    # Calculate accuracy
    accuracy = np.mean(y_true_classes == y_pred_classes)
    
    # Get classification report
    if class_names is None:
        class_names = [f'Class {i}' for i in range(max(max(y_true_classes), max(y_pred_classes)) + 1)]
    
    report = classification_report(y_true_classes, y_pred_classes, 
                                 target_names=class_names, output_dict=True)
    
    # Get confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'y_true_classes': y_true_classes,
        'y_pred_classes': y_pred_classes
    }

def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2_score': r2
    }

def plot_regression_results(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """
    Plot regression results (predicted vs actual)
    """
    fig = go.Figure()
    
    # Scatter plot of predictions vs actual
    fig.add_trace(go.Scatter(
        x=y_true.flatten(),
        y=y_pred.flatten(),
        mode='markers',
        name='Predictions',
        opacity=0.7
    ))
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title='Predicted vs Actual Values',
        xaxis_title='Actual Values',
        yaxis_title='Predicted Values',
        height=400
    )
    
    return fig

def display_model_performance(model, X_test: np.ndarray, y_test: np.ndarray, 
                            is_classification: bool, class_names: List[str] = None):
    """
    Display comprehensive model performance metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    if is_classification:
        # Classification metrics
        metrics = calculate_classification_metrics(y_test, y_pred, class_names)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Classification Metrics")
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            
            # Display classification report
            report_df = pd.DataFrame(metrics['classification_report']).transpose()
            report_df = report_df.round(4)
            st.dataframe(report_df)
        
        with col2:
            st.subheader("Confusion Matrix")
            cm_fig = plot_confusion_matrix(metrics['y_true_classes'], metrics['y_pred_classes'], class_names)
            st.plotly_chart(cm_fig, use_container_width=True, key="confusion_matrix_chart")
    
    else:
        # Regression metrics
        metrics = calculate_regression_metrics(y_test, y_pred)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Regression Metrics")
            st.metric("R² Score", f"{metrics['r2_score']:.4f}")
            st.metric("Mean Squared Error", f"{metrics['mse']:.4f}")
            st.metric("Root Mean Squared Error", f"{metrics['rmse']:.4f}")
            st.metric("Mean Absolute Error", f"{metrics['mae']:.4f}")
        
        with col2:
            st.subheader("Predicted vs Actual")
            reg_fig = plot_regression_results(y_test, y_pred)
            st.plotly_chart(reg_fig, use_container_width=True, key="regression_results_chart")

def create_architecture_diagram(layers: List[int]) -> str:
    """
    Create a text-based visualization of the neural network architecture
    """
    diagram = "Neural Network Architecture:\n"
    diagram += "=" * 50 + "\n\n"
    
    for i, layer_size in enumerate(layers):
        if i == 0:
            layer_type = "Input Layer"
        elif i == len(layers) - 1:
            layer_type = "Output Layer"
        else:
            layer_type = f"Hidden Layer {i}"
        
        diagram += f"{layer_type:>15}: {layer_size} neurons\n"
        
        if i < len(layers) - 1:
            diagram += " " * 15 + "    |\n"
            diagram += " " * 15 + "    v\n"
    
    total_params = 0
    for i in range(len(layers) - 1):
        layer_params = layers[i] * layers[i + 1] + layers[i + 1]  # weights + biases
        total_params += layer_params
    
    diagram += "\n" + "=" * 50
    diagram += f"\nTotal Parameters: {total_params:,}"
    
    return diagram

def validate_dataset(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate the uploaded dataset for common issues
    """
    issues = []
    
    # Check for empty dataset
    if df.empty:
        issues.append("Dataset is empty")
        return False, issues
    
    # Check for minimum number of rows
    if len(df) < 10:
        issues.append("Dataset has too few rows (minimum 10 required)")
    
    # Check for all NaN columns
    all_nan_cols = df.columns[df.isnull().all()].tolist()
    if all_nan_cols:
        issues.append(f"Columns with all missing values: {all_nan_cols}")
    
    # Check for high missing value percentage
    high_missing_cols = []
    for col in df.columns:
        missing_pct = df[col].isnull().sum() / len(df) * 100
        if missing_pct > 80:
            high_missing_cols.append(f"{col} ({missing_pct:.1f}%)")
    
    if high_missing_cols:
        issues.append(f"Columns with >80% missing values: {high_missing_cols}")
    
    # Check for constant columns
    constant_cols = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        issues.append(f"Constant columns (no variation): {constant_cols}")
    
    # Check data types
    if df.select_dtypes(include=[np.number]).empty and df.select_dtypes(include=['object']).empty:
        issues.append("No recognizable data types found")
    
    return len(issues) == 0, issues

def get_recommended_architecture(num_features: int, num_classes: int, 
                               problem_type: str) -> List[int]:
    """
    Get recommended neural network architecture based on dataset characteristics
    """
    if problem_type == 'classification':
        # For classification
        if num_features <= 10:
            # Small dataset
            hidden_size = max(8, num_features * 2)
            return [num_features, hidden_size, num_classes]
        elif num_features <= 50:
            # Medium dataset
            hidden_size1 = max(16, num_features)
            hidden_size2 = max(8, hidden_size1 // 2)
            return [num_features, hidden_size1, hidden_size2, num_classes]
        else:
            # Large dataset
            hidden_size1 = max(32, num_features // 2)
            hidden_size2 = max(16, hidden_size1 // 2)
            hidden_size3 = max(8, hidden_size2 // 2)
            return [num_features, hidden_size1, hidden_size2, hidden_size3, num_classes]
    else:
        # For regression
        output_size = 1
        if num_features <= 10:
            hidden_size = max(8, num_features * 2)
            return [num_features, hidden_size, output_size]
        elif num_features <= 50:
            hidden_size1 = max(16, num_features)
            hidden_size2 = max(8, hidden_size1 // 2)
            return [num_features, hidden_size1, hidden_size2, output_size]
        else:
            hidden_size1 = max(32, num_features // 2)
            hidden_size2 = max(16, hidden_size1 // 2)
            return [num_features, hidden_size1, hidden_size2, output_size]
