import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier
import warnings
import mlflow
import mlflow.sklearn
import shutil
warnings.filterwarnings('ignore')



class MultiOutputModel:
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
        self.df['min_loan_amount'] = self.df['loan_amount']*0.7
        self.df['max_loan_amount'] = self.df['loan_amount']*1.5
    
        self.df = self.df.fillna(self.df.mean())
        return self

    
    def get_xy(self):
        
        input_features =  ['amount_repaid', 'msme_frequent_deposit_amount', 'difference_total',\
                           'insur_amount', 'agent_balance', 'number_of_months']

        labels = ['min_loan_amount', 'max_loan_amount']
        # target features
        X = self.df[input_features]
        Y = self.df[labels]
        #Y = self.df['max_loan_amount']
        
        return X, Y
        
 
    
    def encode_transform_x(self, X):
        cols = X.select_dtypes(include=["object", "datetime"]).columns
        labelencoder = LabelEncoder()
        for col in cols:
            X[col] = labelencoder.fit_transform(X[col])
        enc_X = X
        return enc_X

    
    
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
        # Convert X and Y to ndarrays
        X = X.values
        Y = Y.values
        X_train, X_test, y_train, y_test = train_test_split(X,  Y, test_size=0.15, random_state=42) 

        # Define the model and train or fit it
    
        model = MultiOutputClassifier(XGBClassifier(eval_metric='mlogloss')).fit(X_train, y_train)
        #model = XGBClassifier().fit(X_train, y_train)
        
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


    def save_model(self,model_path, model):
        """
        Saves the model using mlflow
        INPUTS:
            filename: str -> the name of the file to which the model is written to using mlflow
            model: matrix -> the model to be saved
        OUTPUT
            None.
        
        """
        shutil.rmtree(model_path)
        mlflow.sklearn.save_model(model, model_path)
    
    
    
    
    def load_model(self,model_path):
        """
        Loads the model using mlflow
        INPUTS:
            fmodel_path: str -> the name of the pickle file to load
        
        OUTPUT
           loaded_model:  the loaded model
        
        """
        loaded_model =  mlflow.sklearn.load_model(model_path)
        return loaded_model
    

