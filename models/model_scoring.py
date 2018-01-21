from sklearn.metrics import mean_squared_error
import numpy as np
from scipy.stats import rankdata

#pathname for the dataframe with 273 features for 1000 pictures
pathname_feature = "/Users/estelleaflalo/Desktop/ITC/DataSciencesTrack/Project/df_1000.csv"
pathname_label = "/Users/estelleaflalo/Desktop/M2_Data_Science/First_Period/Machine_Learning_from_Theory_to_Practice/Project/Final_Machine_Learning/portrait/dataframes/output_train.csv"


class Scoring():
    def __init__(self, model):
        self.model = model

    def split(self, test_size = 0.33):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        self.pred = self.model.predict(X_test)
        return self.pred

    def evaluate_portraits(self, y_test):
        y_true_rank = rankdata(y_test)
        y_pred_rank = rankdata(self.pred)
        square_distance = np.dot((y_pred_rank - y_true_rank).T, (y_pred_rank - y_true_rank))
        accuracy = 1 - 6 * square_distance / (y_pred_rank.shape[0] * (y_pred_rank.shape[0] ** 2 - 1))
        return accuracy



# scoring on portraits


#predicting scores of wedding pictures


#

