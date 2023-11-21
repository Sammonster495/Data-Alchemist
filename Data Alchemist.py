from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from ydata_profiling import ProfileReport
import pandas as pd
import numpy as np
import json

# Apply the dataset you want to predict an algorithm for
df = pd.read_csv("your-dataset.csv") # replace "your-dataset.csv" with the filename of the dataset

# Data profiling using ProfileReport stored in profile
profile = ProfileReport(df, title='Pandas Profiling Report')

# Profile as JSON file
profile.to_file("your_report.json")

# Load the JSON file into an object data
with open('your_report.json') as f:
    data = json.load(f)

# Classifying the dataset columns into Categorical and Numeric
attributes = {}; regress = {}; classify = {}
best_model = 0.0; best_accuracy = 0.0
for i in df.columns:
    attributes[i] = data['variables'][i]['type']

# Handling missing data through imputation and encoding
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
encoder = LabelEncoder()
for i in attributes:
    if attributes[i] == 'Categorical' and (df[i].isnull().sum() or df[i].dtypes == 'object'):
        df[i] = encoder.fit_transform(df[i])
    elif attributes[i] == 'Numeric' and df[i].isnull().sum():
        df[[i]] = imputer.fit_transform(df[[i]])

# Determining the target set from user input
print(list(df.columns))
target = input("Enter the target variable : ")

# Creating the feature and target sets
x = df.drop([target], axis = 1)
y = df[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 7)

# Applying the best suited algorithm to the given dataset and determining the best machine learning algorithm 
if attributes[target] == 'Numeric':
    pipeline_lin = Pipeline([('scaler1', StandardScaler()),('pca1', PCA()), ('lin', LinearRegression())])
    pipeline_dtr = Pipeline([('scaler2', StandardScaler()),('pca2', PCA()), ('dtr', DecisionTreeRegressor())])
    pipeline_svr = Pipeline([('scaler3', StandardScaler()),('pca3', PCA()), ('svr', SVR())])
    pipeline_las = Pipeline([('scaler4', StandardScaler()),('pca4', PCA()), ('las', Lasso(alpha = 0.5))])
    pipeline_rfr = Pipeline([('scaler5', StandardScaler()),('pca5', PCA()), ('rfr', RandomForestRegressor())])
    pipelines = [pipeline_lin, pipeline_dtr, pipeline_svr, pipeline_las, pipeline_rfr]
    pipe_dict = {0:'Linear Regression', 1:'Decision Tree', 2:'SVR', 3:'Lasso Regression', 4:'Random Forest'}
    for pipe in pipelines:
        pipe.fit(x_train, y_train)
    for i, model in enumerate(pipelines):
        print(f"{pipe_dict[i]} Test Accuracy {model.score(x_test, y_test)}")
        if model.score(x_test, y_test) > best_accuracy:
            best_accuracy = model.score(x_test, y_test)
            best_model = i
        regress[model] = model.score(x_test, y_test)
    print(f"Regressor with best accuracy is {pipe_dict[best_model]}")
else:
    pipeline_log = Pipeline([('scaler1', StandardScaler()),('pca1', PCA()), ('log', LogisticRegression())])
    pipeline_dtc = Pipeline([('scaler2', StandardScaler()),('pca2', PCA()), ('dtc', DecisionTreeClassifier())])
    pipeline_svc = Pipeline([('scaler3', StandardScaler()),('pca3', PCA()), ('svc', SVC())])
    pipeline_nbc = Pipeline([('scaler4', StandardScaler()),('pca4', PCA()), ('nbc', GaussianNB())])
    pipeline_knn = Pipeline([('scaler5', StandardScaler()),('pca5', PCA()), ('knn', KNeighborsClassifier())])
    pipelines = [pipeline_log, pipeline_dtc, pipeline_svc, pipeline_nbc, pipeline_knn]
    pipe_dict = {0:'Logistic Regression', 1:'Decision Tree', 2:'SVC', 3:'Naive Bayes', 4:'K Nearest Neighbors'}
    x, y = SMOTE().fit_resample(x, y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 7)
    for pipe in pipelines:
        pipe.fit(x_train, y_train)
    for i, model in enumerate(pipelines):
        pred = model.predict(x_test)
        print(f"{pipe_dict[i]} Test Accuracy {accuracy_score(y_test, pred)}")
        if accuracy_score(y_test, pred) > best_accuracy:
            best_accuracy = accuracy_score(y_test, pred)
            best_model = i
        regress[model] = model.score(x_test, y_test)
    print(f"Classifier with best accuracy is {pipe_dict[best_model]}")