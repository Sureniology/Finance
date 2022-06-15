import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputRegressor
import pickle
import numpy as np
## Supress warnings
import warnings
warnings.filterwarnings('ignore')



class MultiColumnLabelEncoder:
    """
    A Class of Sklearn Multiple Columns Label Encoder and inverse_transform functions.
    The class provides functions to encode and reverse the encoding of labels.
    
    Source: https://stackoverflow.com/questions/58217005/how-to-reverse-label-encoder-from-sklearn-for-multiple-columns
    """

    def __init__(self, columns=None):
        self.columns = columns # array of column names to encode


    def fit(self, X, y=None):
        self.encoders = {}
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            self.encoders[col] = LabelEncoder().fit(X[col])
        return self


    def transform(self, X):
        output = X.copy()
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            output[col] = self.encoders[col].transform(X[col])
        return output


    def fit_transform(self, X, y=None):
        return self.fit(X,y).transform(X)


    def inverse_transform(self, X):
        output = X.copy()
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            output[col] = self.encoders[col].inverse_transform(X[col])
        return output

class ModelClass:
    """
    A class for the mult-column-labels with the following features:
    1. processing(self, filepath): returns the class attribute' dataframe df.
    2. get_xy(self): Split the dataframe into X and Y
    3. encode_transform_x(self, X): encode the X dataframe
    4. model(self, X, Y): train and fit the model. 
    5. evaluate_multi_output_model(self, X_test, y_test, model): 
    6. save_model(self,filename, model): saves the model
    7. load_model(model_name): load an already saved model
    8. get_input_from_users(self): Get user inputs from the user
    """
    def processing(self, filepath):
        """
        Processes the dataset, get the input and target labels:
        INPUT: 
            filepath -> str-- path to the dataset
        OUTPUTS:
        self:    
        """
        self.df = pd.read_csv(filepath)
        # Get the minimum loan amount by multiplying loan_amount by 0.7 and max_loan_amount by multiplying by 1.5
        self.df['min_loan_amount'] = self.df['loan_amount']*0.5
        self.df['max_loan_amount'] = self.df['loan_amount']*1.5
        self.df[['daily', 'weekly', 'monthly']] = pd.get_dummies(self.df.schedule)
    
        self.df = self.df.fillna(self.df.mean())
        return self

    
    def get_xy(self):
        
        input_features =   ['number_of_months', 'schedule', 'insur_amount','msme_business_description','msme_frequent_deposit_amount',\
                            'msme_full_name', 'msme_sex', 'msme_shop_address','flat_or_reducing_loan_type']
        labels = ['min_loan_amount', 'max_loan_amount']
        # target features
        X = self.df[input_features]
        Y = self.df[labels]
       
        
        return X, Y
        
     
    def encode_transform_x(self, X):
        multi = MultiColumnLabelEncoder(columns=X.columns)
        enc_X = multi.fit_transform(X)
        return enc_X
    
    def encode_transform_y(self, Y):
        multi = MultiColumnLabelEncoder(columns=Y.columns)
        
        return multi.fit_transform(Y)

        
    
    def model(self, X, Y):
        """A function to train a model using MultiOutputClassifie 
             with XGBoostClassifier as an estimator.
    
        INPUT:
        X - input features
        Y - target label
        OUTPUTS:
        X_test - X_test  
        y_test - y_test
        model  - trained model
        """
             # Change Y to integer
        Y = Y.astype('float32')
        # Encode categorical X values
        cols = X.select_dtypes(include=["object", "datetime"]).columns
        for col in cols:
            encoder = LabelEncoder()
            X[col] =encoder.fit_transform(X[col])
        X = X.values
        Y = Y.values
        X_train, X_test, y_train, y_test = train_test_split(X,  Y, test_size=0.20, random_state=42) 

        # Define the model and train or fit it
        #clf = XGBClassifier(eval_metric='mlogloss')
        clf = XGBRegressor()
        #MultiOutputRegressor
        #model = MultiOutputClassifier(clf).fit(X_train, y_train)
        model = MultiOutputRegressor(clf).fit(X_train, y_train)
        #model = XGBClassifier().fit(X_train, y_train)
        #booster = clf.get_booster()
        #booster.save_model('test_model.bin')
        
        return X_test, y_test, model


    def evaluate_model(self, X_test, y_test, model):
        """
        Evaluates the model by displaying the precision, recall, f1_score,
        RMSE, and accuracy of the model
        INPUTS:
            X_test, y_test, model
        OUTPUTS:
            None
    
        """
        y_true = model.predict(X_test)
        print('='*50)
        print('='*50)
        print("The accuracy of the multi-outputs model is: {}".format(round(model.score(X_test, y_test),3)*100))
        print(f"The RMSE error is : {round(mean_squared_error(y_test, y_true, squared = False),4)}")


    def save_model(self,filename, model):
        """
        Saves the model as a pickle file
        INPUTS:
            filename: str -> the name of the file to which the model is written to ending with .pkl
            model: matrix -> the model to be saved
        OUTPUT
            None.
        
        """
        pickle.dump(model, open(filename, 'wb'))
    
    def load_model1(self,filename):
        """
        Saves the model as a binary
        INPUTS:
            filename: str -> the name of the file to which the model is written to with booster
            model: matrix -> the model to be saved
        OUTPUT
            None.
        
        """
       # booster =  xgb.Booster()
       # booster.load_model(filename)
    
    
    def load_model(self,filename):
        """
        Saves the model as a pickle file
        INPUTS:
            filename: str -> the name of the pickle file to load
        
        OUTPUT
           loaded_model:  the loaded model
        
        """
        loaded_model = pickle.load(open(filename, 'rb'))
        return loaded_model
