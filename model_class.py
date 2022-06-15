import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier
import pickle
import numpy as np
import datetime 
## Supress warnings
import warnings
warnings.filterwarnings('ignore')


class MultiColumnLabelModel:
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
    
    
        return self

    
    def get_xy(self):
        
        input_features =  ['amount_repaid', 'transaction_history_id_x',  'schedule', 'frequent_deposit_amount',\
                       'number_of_months',  'status', 'date', 'business_description', 'savings_balance', \
                       'paid_on_time', 'sex', 'msme_insurance_amount']
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
    
        model = MultiOutputClassifier(XGBClassifier()).fit(X_train, y_train)
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
    

    def get_input_from_users(self):
        """Gets inputs from the user. 
        INPUTS:
            None:
        OUTPUT:
            df: a dataframe that would be used for predicting an msme(s)'s loan capacity
    
        """
    
        columns = ['amount_repaid', 'transaction_history_id_x', 'schedule', 'frequent_deposit_amount',\
               'number_of_months', 'status', 'date', 'business_description', 'savings_balance', 'paid_on_time',\
               'sex', 'msme_insurance_amount']

        required_inputs_from_users = ['amount_repaid', 'transaction_history_id_x', 'schedule', 'frequent_deposit_amount',\
               'number_of_months', 'status', 'date', 'business_description', 'savings_balance', 'loan_due_date',\
               'loan_repayment_transaction_date', 'sex', 'msme_insurance_amount']
        try:
            amount_repaid = int(input("Enter the amount of loan repaid by this MSME (Number only):   "))
            while amount_repaid <0:
                amount_repaid = int(input("Enter the amount of loan repaid by this MSME (Must be positive number):   "))
        
            transaction_history_id_x = int(input("Enter the loan record transaction_history_id (Number only):   "))
    
            while transaction_history_id_x < 0:
        
                transaction_history_id_x = int(input("Try again. The loan record transaction_history_id (Numeric positive value):   "))
        
            schedule =   input("What is the schedule of the loan repayment? (Must be one of 'daily', 'weekly' or 'monthly'  )").lower()
    
    
    
            frequent_deposit_amount = int(input("Enter the frequent deposit amount of an MSME. Integer value only:   "))
    
            if frequent_deposit_amount < 0: 
                frequent_deposit_amount = int(input("Enter the frequent deposit amount of an MSME. Integer value only:   "))
    
            number_of_months = int(input("Enter the duration of the loan in months. Integer Value Only:  "))
            while number_of_months < 0: 
                number_of_months = int(input("Enter the duration of the loan in months. Integer Value Only:  "))
    
            status = int(input("Enter the status the loan. It must be 0 for new loan, 1 for in-progress or 2 for repaid:  "))
      
    
            date = input("Enter the date the loan was or is to be taken. The format should be 'YYYY-MM-DD':    ")
    
    
            business_description = input("Enter the msme's business description. Examples: ' tailoring', 'vulcanizer', 'grocery store', 'trader', 'computer center':   ").lower()
    
    
            savings_balance = int(input("Enter the MSME savings balance. Integer Value Only:  "))
    
    
            loan_due_date = input("Enter the loan_due_date. The format should be 'YYYY-MM-DD':    ")
               
        
            loan_repayment_transaction_date  = input("Enter the loan_repayment_transaction_date i.e  date loan was or is to be repiad. The format should be 'YYYY-MM-DD':    ")
    
    
            sex  = input("Enter the Gender of the msme operator. Must be 'male' or 'female':   ").lower()
    
            msme_insurance_amount = int(input("Enter the msme insurance amount. Integer Value Only:  "))
    
    
            # Convert the dates to datetime
            date = pd.to_datetime(date)
            loan_due_date = pd.to_datetime(loan_due_date)
            loan_repayment_transaction_date = pd.to_datetime(loan_repayment_transaction_date)
    
            paid_on_time = np.where((loan_repayment_transaction_date <= loan_due_date), 1,0)
    
    
            values = [amount_repaid, transaction_history_id_x,   schedule, frequent_deposit_amount ,\
               number_of_months, status, date, business_description, savings_balance, paid_on_time,\
               sex, msme_insurance_amount]
    
            data = {}
            for key, value in zip(columns, values):
                data[key] = value
    
            df = pd.DataFrame(data=data, index=[0])
        
    
            return df
    
        except Exception as e:
            print("There is an error. Try again.")
        
   
        
