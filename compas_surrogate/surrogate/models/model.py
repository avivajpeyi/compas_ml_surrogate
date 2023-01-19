from abc import ABC, abstractmethod

import numpy as np


class Model(ABC):
    """Base class for surrogate models."""

    @abstractmethod
    @classmethod
    def load(cls, path: str) -> "Model":
        """Load a model from a file."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save a model to a file."""
        pass

    @abstractmethod
    def train(self, *args, **kwargs) -> None:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the output of the model for the given input."""
        pass

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)
