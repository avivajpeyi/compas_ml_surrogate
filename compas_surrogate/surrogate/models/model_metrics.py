import datetime

from sklearn import metrics


class ModelMetrics:
    @staticmethod
    def current_time():
        return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    def mean_absolute_error(self, data):
        """Return the average of absolute errors of all the data points in the given dataset.
        np.mean(np.abs(pred_y - data[1]))
        """
        true_y = data[1]
        _, pred_y, _ = self(data[0])
        return round(metrics.mean_absolute_error(true_y, pred_y), 2)

    def mean_squared_error(self, data):
        """Return the average of the squares of the errors of all the data points in the given dataset.
        np.mean((pred_y - data[1]) ** 2)
        """
        true_y = data[1]
        _, pred_y, _ = self(data[0])
        return round(metrics.mean_squared_error(true_y, pred_y), 2)

    def r_sqred(self, data):
        """Return the R^2 =  1- relative mean squared error of the model
        This is the coefficient of determination.
        This tells us how well the unknown samples will be predicted by our model.
        The best possible score is 1.0, but the score can be negative as well.
        1 - np.mean((pred_y - true_y) ** 2) / np.var(true_y)
        """
        true_y = data[1]
        _, pred_y, _ = self(data[0])
        return round(metrics.r2_score(true_y, pred_y), 2)

    def _get_metrics(self, data):
        return Metrics(
            self.mean_absolute_error(data),
            self.mean_squared_error(data),
            self.r_sqred(data),
        )

    def train_test_metrics(self, train_data, test_data):
        return {
            "train": self._get_metrics(train_data),
            "test": self._get_metrics(test_data),
        }


class Metrics:
    def __init__(self, mae, mse, r2):
        self.mae = mae
        self.mse = mse
        self.r2 = r2

    def __str__(self):
        return f"MAE: {self.mae}, MSE: {self.mse}, R2: {self.r2}"

    def __dict__(self):
        return {"MAE": self.mae, "MSE": self.mse, "R2": self.r2}

    def __repr__(self):
        return self.__str__()
