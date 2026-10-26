import numpy as np
import json
import os
from typing import List, Tuple, Optional, Dict, Any

class VnexAI:
    """
    Custom Neural Network implementation from scratch using NumPy
    """
    
    def __init__(self, layers: List[int], learning_rate: float = 0.01, 
                 activation: str = 'relu', output_activation: str = 'softmax'):
        """
        Initialize the VnexAI neural network
        
        Args:
            layers: List of integers representing the number of neurons in each layer
            learning_rate: Learning rate for gradient descent
            activation: Activation function for hidden layers ('relu', 'sigmoid', 'tanh')
            output_activation: Activation function for output layer ('softmax', 'sigmoid', 'linear')
        """
        self.layers = layers
        self.learning_rate = learning_rate
        self.activation = activation
        self.output_activation = output_activation
        self.weights = []
        self.biases = []
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # Initialize weights and biases
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        """Initialize weights and biases using Xavier initialization"""
        np.random.seed(42)  # For reproducibility
        
        for i in range(len(self.layers) - 1):
            # Xavier initialization
            weight = np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2.0 / self.layers[i])
            bias = np.zeros((1, self.layers[i + 1]))
            
            self.weights.append(weight)
            self.biases.append(bias)
    
    def _activation_function(self, x: np.ndarray, activation: str) -> np.ndarray:
        """Apply activation function"""
        if activation == 'relu':
            return np.maximum(0, x)
        elif activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -250, 250)))  # Clip to prevent overflow
        elif activation == 'tanh':
            return np.tanh(x)
        elif activation == 'softmax':
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Numerical stability
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        elif activation == 'linear':
            return x
        else:
            raise ValueError(f"Unknown activation function: {activation}")
    
    def _activation_derivative(self, x: np.ndarray, activation: str) -> np.ndarray:
        """Compute derivative of activation function"""
        if activation == 'relu':
            return (x > 0).astype(float)
        elif activation == 'sigmoid':
            s = self._activation_function(x, 'sigmoid')
            return s * (1 - s)
        elif activation == 'tanh':
            t = self._activation_function(x, 'tanh')
            return 1 - t ** 2
        elif activation == 'linear':
            return np.ones_like(x)
        else:
            raise ValueError(f"Unknown activation function: {activation}")
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        Forward propagation
        
        Returns:
            output: Final output of the network
            activations: List of activations for each layer
            z_values: List of pre-activation values for each layer
        """
        activations = [X]
        z_values = []
        
        current_input = X
        
        for i in range(len(self.weights)):
            # Linear transformation
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            z_values.append(z)
            
            # Apply activation function
            if i == len(self.weights) - 1:  # Output layer
                activation = self._activation_function(z, self.output_activation)
            else:  # Hidden layers
                activation = self._activation_function(z, self.activation)
            
            activations.append(activation)
            current_input = activation
        
        return activations[-1], activations, z_values
    
    def backward(self, X: np.ndarray, y: np.ndarray, activations: List[np.ndarray], 
                z_values: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Backward propagation
        
        Returns:
            weight_gradients: Gradients for weights
            bias_gradients: Gradients for biases
        """
        m = X.shape[0]  # Number of samples
        weight_gradients = []
        bias_gradients = []
        
        # Convert y to one-hot if necessary
        if len(y.shape) == 1 and self.output_activation == 'softmax':
            y_one_hot = np.zeros((y.shape[0], self.layers[-1]))
            y_one_hot[np.arange(y.shape[0]), y] = 1
            y = y_one_hot
        
        # Initialize error for output layer
        if self.output_activation == 'softmax':
            # For softmax with cross-entropy, derivative is simplified
            delta = activations[-1] - y
        else:
            # For other activations, compute derivative
            output_derivative = self._activation_derivative(z_values[-1], self.output_activation)
            delta = (activations[-1] - y) * output_derivative
        
        # Backpropagate through all layers
        for i in range(len(self.weights) - 1, -1, -1):
            # Compute gradients
            weight_grad = np.dot(activations[i].T, delta) / m
            bias_grad = np.sum(delta, axis=0, keepdims=True) / m
            
            weight_gradients.insert(0, weight_grad)
            bias_gradients.insert(0, bias_grad)
            
            # Compute delta for previous layer (if not input layer)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self._activation_derivative(z_values[i-1], self.activation)
        
        return weight_gradients, bias_gradients
    
    def update_parameters(self, weight_gradients: List[np.ndarray], bias_gradients: List[np.ndarray]):
        """Update weights and biases using gradients"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]
    
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute loss based on output activation"""
        m = y_true.shape[0]
        
        if self.output_activation == 'softmax':
            # Cross-entropy loss
            if len(y_true.shape) == 1:
                # Convert to one-hot
                y_one_hot = np.zeros((y_true.shape[0], self.layers[-1]))
                y_one_hot[np.arange(y_true.shape[0]), y_true] = 1
                y_true = y_one_hot
            
            # Clip predictions to prevent log(0)
            y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
            loss = -np.sum(y_true * np.log(y_pred_clipped)) / m
        else:
            # Mean squared error for other activations
            loss = np.mean((y_true - y_pred) ** 2)
        
        return loss
    
    def compute_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute accuracy"""
        if self.output_activation == 'softmax':
            if len(y_true.shape) > 1:
                y_true = np.argmax(y_true, axis=1)
            y_pred_classes = np.argmax(y_pred, axis=1)
        else:
            # For regression or binary classification
            if self.layers[-1] == 1:
                y_pred_classes = (y_pred > 0.5).astype(int).flatten()
                if len(y_true.shape) > 1:
                    y_true = y_true.flatten()
            else:
                y_pred_classes = np.round(y_pred)
        
        return np.mean(y_true == y_pred_classes)
    
    def train_batch(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Train on a single batch"""
        # Forward pass
        output, activations, z_values = self.forward(X)
        
        # Compute loss and accuracy
        loss = self.compute_loss(y, output)
        accuracy = self.compute_accuracy(y, output)
        
        # Backward pass
        weight_gradients, bias_gradients = self.backward(X, y, activations, z_values)
        
        # Update parameters
        self.update_parameters(weight_gradients, bias_gradients)
        
        return loss, accuracy
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        output, _, _ = self.forward(X)
        return output
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            epochs: int = 100, batch_size: int = 32, verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the neural network
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size for mini-batch gradient descent
            verbose: Whether to print training progress
        
        Returns:
            Training history dictionary
        """
        n_samples = X_train.shape[0]
        n_batches = max(1, n_samples // batch_size)
        
        # Reset training history
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_losses = []
            epoch_accuracies = []
            
            # Mini-batch training
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                batch_loss, batch_accuracy = self.train_batch(X_batch, y_batch)
                epoch_losses.append(batch_loss)
                epoch_accuracies.append(batch_accuracy)
            
            # Compute average epoch metrics
            avg_loss = np.mean(epoch_losses)
            avg_accuracy = np.mean(epoch_accuracies)
            
            self.training_history['loss'].append(avg_loss)
            self.training_history['accuracy'].append(avg_accuracy)
            
            # Validation metrics
            if X_val is not None and y_val is not None:
                val_pred = self.predict(X_val)
                val_loss = self.compute_loss(y_val, val_pred)
                val_accuracy = self.compute_accuracy(y_val, val_pred)
                
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_accuracy'].append(val_accuracy)
                
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {avg_accuracy:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_accuracy:.4f}")
            else:
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {avg_accuracy:.4f}")
        
        return self.training_history
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'layers': self.layers,
            'learning_rate': self.learning_rate,
            'activation': self.activation,
            'output_activation': self.output_activation,
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases],
            'training_history': self.training_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        self.layers = model_data['layers']
        self.learning_rate = model_data['learning_rate']
        self.activation = model_data['activation']
        self.output_activation = model_data['output_activation']
        self.weights = [np.array(w) for w in model_data['weights']]
        self.biases = [np.array(b) for b in model_data['biases']]
        self.training_history = model_data['training_history']
    
    def get_model_summary(self) -> str:
        """Get a summary of the model architecture"""
        total_params = sum(w.size + b.size for w, b in zip(self.weights, self.biases))
        
        summary = f"VnexAI Neural Network Summary\n"
        summary += f"{'='*40}\n"
        summary += f"Architecture: {' -> '.join(map(str, self.layers))}\n"
        summary += f"Activation: {self.activation} (hidden), {self.output_activation} (output)\n"
        summary += f"Learning Rate: {self.learning_rate}\n"
        summary += f"Total Parameters: {total_params:,}\n"
        summary += f"{'='*40}\n"
        
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            summary += f"Layer {i+1}: {w.shape[0]} -> {w.shape[1]} (params: {w.size + b.size})\n"
        
        return summary
