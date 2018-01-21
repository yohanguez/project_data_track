from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#pathname for the dataframe with 273 features for 1000 pictures
pathname_feature = "/Users/estelleaflalo/Desktop/ITC/DataSciencesTrack/Project/df_1000.csv"
pathname_label = "/Users/estelleaflalo/Desktop/M2_Data_Science/First_Period/Machine_Learning_from_Theory_to_Practice/Project/Final_Machine_Learning/portrait/dataframes/output_train.csv"


class Scoring():
    def __init__(self, X, y, model):
        self.X = X
        self.y = y
        self.model = model

    def split(self, test_size = 0.33):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def fit_predict(self):
        self.model.fit(self.X_train, self.y_train)
        self.pred = self.model.predict(self.X_test)
        return self.pred

    def evaluate(self):
        return mean_squared_error(self.test, self.pred)



# scoring on portraits


#predicting scores of wedding pictures


#

