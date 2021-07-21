import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from pydataset import data
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#import pandas as pd

st.header('Try end to end predictive modeling on different datasets')

st.info("""
- Pick the dataset
- Validate the dataset
- Prepare the dataset (Impute/Scaling/Categorical Encoding/Imbalance)
- Pick the Machine Learning Algorithmn
- Analyse the results (Accuracy, MAE, Recall, Precision, F1)
""")

class MlStreamlitApp:

    def __init__(self):
        self.dataset_list = data()
    #, show_doc=True
    @st.cache(suppress_st_warning=True)
    def load_data(self,dataset_name):
        df = data(str(dataset_name))
        df.columns = df.columns.str.replace('.','_')
        df.columns = df.columns.str.lower()
        return df

    '''
    @st.cache(suppress_st_warning=True)
    def show_dataset_doc(self,dataset_name):
        st.write(dataset_name)
        st.code(data('iris',show_doc=True))
    '''

    def show_datasets_list(self):
        st.info('Datasets details')
        st.write(self.dataset_list)

    def run(self):
        st.sidebar.title('Streamlit ML App')
        dataset_list=['']+list(self.dataset_list.dataset_id)        
        dataset_name = st.sidebar.selectbox('Select the Dataset',dataset_list)
        process_selection = st.sidebar.radio("What on your mind?",('EDA', 'Predictive Modelling'))

        if dataset_name == '':
            st.sidebar.warning('Select the Dataset')
            self.show_datasets_list()
        elif (dataset_name and process_selection == 'Predictive Modelling'):
            df = self.load_data(dataset_name)
            st.write(df.head())
            #image = Image.open('./ml_process.jpeg')
            #st.sidebar.image(image)
            dataset_target = st.selectbox('Select the Target', list(df.columns))
            df=df.rename(columns={dataset_target:'target'})

            df_dum=pd.get_dummies(df.loc[:, df.columns != 'target'],drop_first=True)
            df=pd.concat([df_dum,df.target],axis=1)

            #algo_type = st.selectbox('Classification or Regression', list(['Classification','Regression']))

            if df.target.dtypes == 'object':
                algo_type='Classification'
                ml_algos = ['LogisticRegression','DecisionTreeClassifier','RandomForestClassifier','AdaBoostClassifier']
            else:
                algo_type='Regression'
                ml_algos = ['LinearRegression']

            # if algo_type == 'Classification':
            #     ml_algos = ['LogisticRegression','DecisionTreeClassifier','RandomForestClassifier','AdaBoostClassifier']

            # else:
            #     ml_algos = ['LinearRegression']

            X= df.loc[:, df.columns != 'target']
            y= df['target']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

            #st.write(X_train.head())
            #st.write(y_test.head())

            ml_algo = st.selectbox('Select the ML Algo', list(ml_algos))

            if ml_algo == 'LogisticRegression':
                clf_fit = LogisticRegression().fit(X_train, y_train)
                predictions = clf_fit.predict(X_test)
                st.write(predictions[1:5])
            elif ml_algo == 'DecisionTreeClassifier':
                clf_fit = DecisionTreeClassifier().fit(X_train, y_train)
                predictions = clf_fit.predict(X_test)
                st.write(predictions[1:5])
                #RandomForestClassifier
            elif ml_algo == 'RandomForestClassifier':
                clf_fit = RandomForestClassifier().fit(X_train, y_train)
                predictions = clf_fit.predict(X_test)
                st.write(predictions[1:5])
            elif ml_algo == 'AdaBoostClassifier':
                clf_fit = AdaBoostClassifier().fit(X_train, y_train)
                predictions = clf_fit.predict(X_test)
                st.write(predictions[1:5])
            elif ml_algo == 'LinearRegression':
                clf_fit = LinearRegression().fit(X_train, y_train)
                predictions = clf_fit.predict(X_test)
                st.write(predictions[1:5])           
            else:
                st.write('No ML Algo selected')

            if algo_type=='Classification':
                st.write("""
                    Confusion Matrix
                """)
                st.write(confusion_matrix(y_test, predictions))
                st.write("""
                #### Accuracy Score:
                """)
                st.write( accuracy_score(y_test, predictions))
                st.write("""
                #### Other Scores - precision_recall_fscore:
                """)
                precision,recall,f1_score,support=precision_recall_fscore_support(y_test, predictions,average='weighted')
                st.write(round(precision,2),round(recall,2),round(f1_score,2))
            else:
                st.write("""  ### Model Evaluation   """)
                r2_metrics = r2_score(y_test, predictions)
                mse = mean_squared_error(y_test, predictions)
                rmse = np.sqrt(mse)
                mae=mean_absolute_error(y_test, predictions)
                st.write("""  #### Rsquared, MSE , RMSE, MAE """)
                st.write(round(r2_metrics,2),round(mse,2),round(rmse,2),round(mae,2))
        else:
            eda_selection = st.sidebar.radio("What you want to see?",('Summary', 'Plots'))
            df = self.load_data(dataset_name)
            if eda_selection == 'Summary':
                st.write("Glimpse of Data",df.head(10))
                st.write("Total No. of Rows",df.shape[0])
                st.write("Total No. of Columns",df.shape[1])
                st.write("Types of Columns",df.dtypes)
                st.write("Summary Stats",df.describe().T)
                st.write("Total Nulls in the columns",df.isnull().sum())
                st.write("Total Duplicate Rows",df[df.duplicated()].shape[0])
                st.write("Correlation Matrix",df.corr())
            else:
                st.info('Plots') #WIP


if __name__ == '__main__':
    mlApp = MlStreamlitApp()
    mlApp.run()