from abc import ABC
from typing import Optional, Dict, Any, Union, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import joblib
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from .base import BaseModel

@dataclass
class SklearnModel(BaseModel):
    """
    Base wrapper for Scikit-Learn models to standardize BCI experiment workflows.

    This class handles the entire lifecycle of a Scikit-Learn model, including:
    - Automatic Label Encoding (string labels -> integers).
    - Training and validation logic.
    - Metric calculation flattened for MLflow logging.
    - Model persistence (saving/loading).

    Attributes:
        model_class (type[BaseEstimator]): The Scikit-Learn class to instantiate (e.g., SVC).
        params (dict): Dictionary of hyperparameters to pass to the model constructor.
        model (BaseEstimator): The instantiated Scikit-Learn model object.
        encoder (LabelEncoder): Encoder to handle string-to-integer label transformations.
    """
    model_class: type[BaseEstimator]
    params: dict = field(default_factory=dict)
    model: BaseEstimator = field(init=False)
    encoder: LabelEncoder = field(default_factory=LabelEncoder, init=False)
    
    def __post_init__(self):
        """Initializes the model instance with the provided parameters."""
        self._is_trained = False
        self.model = self.model_class(**self.params)

    def train(self, 
              x_train: np.ndarray,
              y_train: np.ndarray,
              validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, float]:
        """
        Trains the model on the provided data and calculates performance metrics.

        This method automatically fits a LabelEncoder on `y_train` to handle string labels.

        Args:
            x_train (np.ndarray): Training features of shape (n_samples, n_features).
            y_train (np.ndarray): Training labels (can be strings or integers).
            validation_data (tuple, optional): A tuple (x_val, y_val) for validation. 
                                             Defaults to None.

        Returns:
            Dict[str, float]: A dictionary of flattened metrics (e.g., 'train_accuracy', 
                              'val_f1_macro') suitable for MLflow logging.
        """
        
        # Data Pipeline: Process features if necessary
        x_train = self._process_data(x_train)
        
        # Label Encoding: Fit the encoder ONLY on training data
        # Transforms labels (e.g., 'left_hand' -> 0) and stores the mapping
        y_train_enc = self.encoder.fit_transform(y_train)
        
        # Model Training
        self.model.fit(x_train, y_train_enc)
        self._is_trained = True

        # Training Metrics
        # Predict using the trained model on training data
        train_pred_enc = self.model.predict(x_train)
        
        # Calculate and flatten metrics
        metrics = self._calculate_metrics(y_train_enc, train_pred_enc, prefix='train_')
        
        # Validation Metrics
        if validation_data:
            x_val, y_val = validation_data
            # Evaluate on validation set using the same encoder
            _, _, val_metrics = self.evaluate(x_val, y_val, prefix='val_')
            metrics.update(val_metrics)
            
        return metrics

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts class labels (encoded as integers) for samples in x.

        Args:
            x (np.ndarray): Input features of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted integer labels.

        Raises:
            ValueError: If the model has not been trained yet.
        """
        if not self._is_trained:
            raise ValueError("The model must be trained first.")
        
        x = self._process_data(x)
        return self.model.predict(x)

    def predict_labels(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts class labels and returns them as their original string representation.

        Args:
            x (np.ndarray): Input features.

        Returns:
            np.ndarray: Predicted string labels (e.g., ['left', 'right']).
        """
        pred_idxs = self.predict(x)
        return self.encoder.inverse_transform(pred_idxs)

    def evaluate(self, 
                 x_test: np.ndarray, 
                 y_test: np.ndarray,
                 prefix: str = ""
                 ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Evaluates the model on a test set.

        Args:
            x_test (np.ndarray): Test features.
            y_test (np.ndarray): True test labels (strings or integers).
            prefix (str, optional): Prefix for metric keys (e.g., 'test_'). Defaults to "".

        Returns:
            Tuple containing:
                - y_test_enc (np.ndarray): Encoded ground truth labels.
                - y_pred_enc (np.ndarray): Encoded predicted labels.
                - metrics (Dict[str, float]): Dictionary of performance metrics.

        Raises:
            ValueError: If unseen labels appear in `y_test` that were not in `y_train`.
        """
        
        if not self._is_trained:
            raise ValueError("Model must be trained before evaluation")
            
        x_test = self._process_data(x_test)
        
        # Transform y_test using the already trained encoder.
        try:
            y_test_enc = self.encoder.transform(y_test)
        except ValueError as e:
            # This happens if y_test contains a label that was not present in y_train
            print(f"Warning: Unknown labels encountered in test set. Error: {e}")
            raise e

        y_pred_enc = self.model.predict(x_test)

        # Calculate metrics using numerical indices
        metrics = self._calculate_metrics(y_test_enc, y_pred_enc, prefix=prefix)

        return y_test_enc, y_pred_enc, metrics

    def _calculate_metrics(self,
                           y_true: np.ndarray,
                           y_pred: np.ndarray,
                           prefix: str = ''
                           ) -> Dict[str, float]:
        """
        Calculates classification metrics and flattens the structure for MLflow.

        It computes Accuracy, Macro F1/Precision/Recall, and Weighted F1/Precision/Recall.

        Args:
            y_true (np.ndarray): Encoded true labels.
            y_pred (np.ndarray): Encoded predicted labels.
            prefix (str, optional): Prefix to add to metric keys.

        Returns:
            Dict[str, float]: Flattened dictionary, e.g., {'test_f1_macro': 0.85}.
        """
        # Generate classification report as a dictionary
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        flat_metrics = {}
        
        # Extract global accuracy
        flat_metrics[f'{prefix}accuracy'] = report['accuracy']
        
        # Extract averages (macro and weighted are standard for multi-class BCI)
        for avg_type in ['macro avg', 'weighted avg']:
            clean_name = avg_type.replace(' avg', '') # becomes 'macro' or 'weighted'
            for metric, value in report[avg_type].items():
                if metric != 'support': # Support is just the count, not a performance metric
                    flat_metrics[f'{prefix}{metric}_{clean_name}'] = value
                    
        return flat_metrics

    def save(self, path: Union[str, Path], overwrite: bool = False) -> None:
        """
        Saves the model state, including the trained model and the label encoder.

        Args:
            path (Union[str, Path]): Destination path for the .joblib file.
            overwrite (bool, optional): Whether to overwrite if file exists. Defaults to False.

        Raises:
            FileExistsError: If file exists and overwrite is False.
        """
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"File {path} already exists")
            
        # Save model, encoder, and training state
        save_data = {
            'model': self.model,
            'encoder': self.encoder,
            'is_trained': self._is_trained,
            'params': self.params
        }
        joblib.dump(save_data, path)

    def _process_data(self, x: np.ndarray) -> np.ndarray:
        """
        Preprocesses input data before feeding it to the model.
        
        Args:
            x (np.ndarray): Raw input array.
            
        Returns:
            np.ndarray: Processed array (currently just ensures numpy format).
        """
        return np.asarray(x)

# --- Specific Model Implementations ---

class SVMModel(SklearnModel):
    """
    Wrapper for Support Vector Classifier (SVC).
    """
    def __init__(self, params: dict):
        super().__init__(model_class=SVC, params=params)

class LogisticModel(SklearnModel):
    """
    Wrapper for Logistic Regression.
    """
    def __init__(self, params: dict):
        super().__init__(model_class=LogisticRegression, params=params)

class LDAModel(SklearnModel):
    """
    Wrapper for Linear Discriminant Analysis (LDA).
    
    Note:
        LDA is considered the 'Gold Standard' for Riemannian Tangent Space classification
        in BCI due to its robustness in high-dimensional spaces when shrinkage is used.
    """
    def __init__(self, params: dict):
        super().__init__(model_class=LinearDiscriminantAnalysis, params=params)

class RidgeModel(SklearnModel):
    """
    Wrapper for Ridge Classifier.
    
    Note:
        Ridge is often faster than Linear SVM for high-dimensional data while 
        providing similar performance.
    """
    def __init__(self, params: dict):
        super().__init__(model_class=RidgeClassifier, params=params)