import streamlit as st
import pandas as pd
import numpy as np
from pydataset import data
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#import pandas as pd

st.beta_container()

st.write("""
## Try end to end predictive modeling on different datasets
- Pick the dataset
- Validate the dataset
- Prepare the dataset (Impute/Scaling/Categorical Encoding/Imbalance)
- Pick the Machine Learning Algorithmn
- Analyse the results (Accuracy, MAE, Recall, Precision, F1)
""")


#st.bar_chart(iris)
dataset_list = data()
dataset_name = st.selectbox('Select the Dataset', list(dataset_list.dataset_id))

st.write(str(dataset_name))

df = data(str(dataset_name))
df.columns = df.columns.str.replace('.','_')
df.columns = df.columns.str.lower()
st.write(df.head())

dataset_target = st.selectbox('Select the Target', list(df.columns))

st.write(dataset_target)

df=df.rename(columns={dataset_target:'target'})

st.write(df.head())

df_dum=pd.get_dummies(df.loc[:, df.columns != 'target'],drop_first=True)
st.write(df_dum.head())
st.write(df.target.head())
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
    st.write("""  ### Model Evaluation   """)
    r2_metrics = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2_metrics,mse,rmse
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
    pass

# if dataset_name == 'iris':
#     iris = data('iris')
#     iris.columns= iris.columns.str.replace('.','_')
#     iris.columns = iris.columns.str.lower()
#     st.write(iris.head())
# elif dataset_name == 'titanic':
#      titanic = data('titanic')
#      st.write(titanic.head())         


