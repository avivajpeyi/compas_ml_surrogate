from abc import ABC, abstractmethod

import numpy as np


class Model(ABC):
    """Base class for surrogate models."""

    @classmethod
    @abstractmethod
    def load(cls, savedir: str) -> "Model":
        """Load a model from a dir."""
        pass

    @abstractmethod
    def save(self, savedir: str) -> None:
        """Save a model to a dir."""
        pass

    @abstractmethod
    def train(self, inputs: np.ndarray, outputs: np.ndarray) -> None:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the output of the model for the given input."""
        pass

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)
