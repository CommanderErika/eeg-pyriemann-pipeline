from abc import ABC, abstractmethod
from typing import Optional
from pathlib import Path

import numpy as np

class BaseModel(ABC):
    @abstractmethod
    def train(self, 
              x_train: np.ndarray, 
              y_train: np.ndarray, 
              validation_data: Optional[tuple[np.ndarray, np.ndarray]] = None) -> None:
        """
        Train the model on the given data.
        
        Args:
            x_train: Training features
            y_train: Training labels
            validation_data: Optional tuple of (validation features, validation labels)
            
        Returns:
            Dictionary of training metrics
        """
        pass

    @abstractmethod
    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> tuple[list, dict[str, float]]:
        """
        Evaluate the model on test data.
        
        Args:
            x_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            x: Input features
            return_proba: Whether to return class probabilities (for classifiers)
            
        Returns:
            Array of predictions or probabilities
        """
        pass

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save the model
            overwrite: Whether to overwrite existing files
        """
        pass