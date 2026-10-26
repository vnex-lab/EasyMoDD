import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any, Optional, List

class DataPreprocessor:
    """
    Data preprocessing utilities for VnexAI
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.target_name = ""
        self.is_classification = True
        self.num_classes = 0
        
    def detect_problem_type(self, y: pd.Series) -> str:
        """
        Automatically detect if the problem is classification or regression
        """
        if y.dtype == 'object' or y.nunique() <= 20:
            return 'classification'
        else:
            return 'regression'
    
    def detect_feature_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Automatically detect feature types in the dataset
        """
        numerical_features = []
        categorical_features = []
        
        for column in df.columns:
            if df[column].dtype in ['int64', 'float64']:
                # Check if it's actually categorical (few unique values)
                if df[column].nunique() <= 10 and df[column].min() >= 0:
                    categorical_features.append(column)
                else:
                    numerical_features.append(column)
            else:
                categorical_features.append(column)
        
        return {
            'numerical': numerical_features,
            'categorical': categorical_features
        }
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        """
        df = df.copy()
        
        for column in df.columns:
            if df[column].isnull().sum() > 0:
                if df[column].dtype in ['int64', 'float64']:
                    # Fill numerical columns with median
                    df[column] = df[column].fillna(df[column].median())
                else:
                    # Fill categorical columns with mode
                    df[column] = df[column].fillna(df[column].mode()[0] if not df[column].mode().empty else 'Unknown')
        
        return df
    
    def encode_categorical_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features using one-hot encoding for features with few categories
        and label encoding for features with many categories
        """
        X = X.copy()
        feature_types = self.detect_feature_types(X)
        
        for column in feature_types['categorical']:
            unique_values = X[column].nunique()
            
            if unique_values <= 10:  # Use one-hot encoding
                if fit:
                    encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
                    encoded_data = encoder.fit_transform(X[[column]])
                    self.encoders[column] = encoder
                    
                    # Create column names for one-hot encoded features
                    feature_names = [f"{column}_{cat}" for cat in encoder.categories_[0][1:]]
                else:
                    encoder = self.encoders[column]
                    encoded_data = encoder.transform(X[[column]])
                    feature_names = [f"{column}_{cat}" for cat in encoder.categories_[0][1:]]
                
                # Create DataFrame with encoded features
                encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=X.index)
                
                # Drop original column and add encoded columns
                X = X.drop(columns=[column])
                X = pd.concat([X, encoded_df], axis=1)
                
            else:  # Use label encoding for high cardinality features
                if fit:
                    encoder = LabelEncoder()
                    X[column] = encoder.fit_transform(X[column].astype(str))
                    self.encoders[column] = encoder
                else:
                    encoder = self.encoders[column]
                    # Handle unknown categories
                    X[column] = X[column].astype(str)
                    mask = X[column].isin(encoder.classes_)
                    X.loc[~mask, column] = encoder.classes_[0]  # Assign to first class for unknown values
                    X[column] = encoder.transform(X[column])
        
        return X
    
    def scale_numerical_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler
        """
        X = X.copy()
        feature_types = self.detect_feature_types(X)
        
        for column in feature_types['numerical']:
            if column in X.columns:  # Check if column still exists after encoding
                if fit:
                    scaler = StandardScaler()
                    X[column] = scaler.fit_transform(X[[column]])
                    self.scalers[column] = scaler
                else:
                    scaler = self.scalers[column]
                    X[column] = scaler.transform(X[[column]])
        
        return X
    
    def encode_target(self, y: pd.Series, fit: bool = True) -> np.ndarray:
        """
        Encode target variable based on problem type
        """
        if self.is_classification:
            if fit:
                encoder = LabelEncoder()
                y_encoded = encoder.fit_transform(y)
                self.encoders['target'] = encoder
                self.num_classes = len(encoder.classes_)
            else:
                encoder = self.encoders['target']
                y_encoded = encoder.transform(y)
        else:
            # For regression, just convert to numpy array
            y_encoded = y.values
            if fit:
                self.num_classes = 1
        
        return y_encoded
    
    def preprocess_data(self, df: pd.DataFrame, target_column: str, 
                       test_size: float = 0.2, val_size: float = 0.1, 
                       random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                       np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete data preprocessing pipeline
        
        Args:
            df: Input dataframe
            target_column: Name of target column
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            random_state: Random state for reproducibility
        
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # Store metadata
        self.target_name = target_column
        self.feature_names = [col for col in df.columns if col != target_column]
        
        # Separate features and target
        X = df[self.feature_names].copy()
        y = df[target_column].copy()
        
        # Detect problem type
        problem_type = self.detect_problem_type(y)
        self.is_classification = (problem_type == 'classification')
        
        # Handle missing values
        X = self.handle_missing_values(X)
        if y.isnull().sum() > 0:
            if self.is_classification:
                y = y.fillna(y.mode()[0] if not y.mode().empty else 'Unknown')
            else:
                y = y.fillna(y.median())
        
        # Split data first
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y if self.is_classification else None
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=random_state,
            stratify=y_temp if self.is_classification else None
        )
        
        # Fit preprocessing on training data
        X_train = self.encode_categorical_features(X_train, fit=True)
        X_train = self.scale_numerical_features(X_train, fit=True)
        y_train_encoded = self.encode_target(y_train, fit=True)
        
        # Transform validation and test data
        X_val = self.encode_categorical_features(X_val, fit=False)
        X_val = self.scale_numerical_features(X_val, fit=False)
        y_val_encoded = self.encode_target(y_val, fit=False)
        
        X_test = self.encode_categorical_features(X_test, fit=False)
        X_test = self.scale_numerical_features(X_test, fit=False)
        y_test_encoded = self.encode_target(y_test, fit=False)
        
        # Update feature names after encoding
        self.feature_names = X_train.columns.tolist()
        
        return (X_train.values, X_val.values, X_test.values, 
                y_train_encoded, y_val_encoded, y_test_encoded)
    
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """
        Get information about the preprocessing steps applied
        """
        info = {
            'problem_type': 'classification' if self.is_classification else 'regression',
            'num_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'num_classes': self.num_classes,
            'scalers_applied': len(self.scalers),
            'encoders_applied': len(self.encoders)
        }
        
        return info
    
    def inverse_transform_target(self, y_encoded: np.ndarray) -> np.ndarray:
        """
        Inverse transform the target variable to original form
        """
        if self.is_classification and 'target' in self.encoders:
            if len(y_encoded.shape) > 1 and y_encoded.shape[1] > 1:
                # If one-hot encoded, convert to class indices first
                y_encoded = np.argmax(y_encoded, axis=1)
            return self.encoders['target'].inverse_transform(y_encoded.astype(int))
        else:
            return y_encoded
