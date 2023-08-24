import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


# Custom callback to save R^2 scores during training
class R2Callback(BaseEstimator):
    def __init__(self):
        self.r2_scores = []

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.validation_data[0])
        r2 = r2_score(self.validation_data[1], y_pred)
        self.r2_scores.append(r2)


# Generate some sample data
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel()

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Gaussian Process Regressor with an RBF kernel
kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
gp = GaussianProcessRegressor(kernel=kernel)

# Create the custom callback
r2_callback = R2Callback()

# Train the model using callbacks
gp.fit(X_train, y_train, validation_data=(X_val, y_val), callbacks=[r2_callback])

# Plotting the R^2 scores over the training period
plt.plot(range(1, len(r2_callback.r2_scores) + 1), r2_callback.r2_scores, marker="o")
plt.xlabel("Epoch")
plt.ylabel("R^2 Score")
plt.title("R^2 Score During Training")
plt.show()
