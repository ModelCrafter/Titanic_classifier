# call moudels
import numpy as np
from pathlib import Path
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.base import BaseEstimator , TransformerMixin 
from sklearn.cluster import KMeans
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer , IterativeImputer
from sklearn.preprocessing import OrdinalEncoder , OneHotEncoder , StandardScaler 
from sklearn.compose import  ColumnTransformer , make_column_selector 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier , Lasso
from sklearn.model_selection import cross_val_score , cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline , make_pipeline

# load data
def load_titanc_csv() ->  tuple[pd.DataFrame , pd.DataFrame]:
    traball_path = Path('datasets/titanic.tgz')
    if  not traball_path.exists():
        print('in if')
        Path('datasets').mkdir(parents=True, exist_ok=True)
        url = 'https://homl.info/titanic.tgz'
        urllib.request.urlretrieve(url , traball_path)
        with tarfile.open(traball_path) as titanic:
            titanic.extractall(path='datasets')
    csv_file_1 = Path('datasets/titanic/train.csv')
    csv_file_2 = Path('datasets/titanic/test.csv')
    return pd.read_csv(csv_file_1) , pd.read_csv(csv_file_2)

# splite data
train , test = load_titanc_csv()
cabin_ = train['Cabin']

train = train.drop(['Name' ,'PassengerId' ,  'Ticket'] , axis=1)
x_test = test.drop(['Name' ,'PassengerId' , 'Cabin' , 'Ticket'] , axis=1)

x_train , y_train = train.drop('Survived' , axis=1) , train['Survived']
some_data = x_train.loc[0]


#some anlyasis
(
#print(train.info())
#print(train.describe())
#print(train.isna().sum()) # will return how many missing value in columns
)

# create pipe columns

class ClusterSimilary(BaseEstimator , TransformerMixin): # cluster to combine all data to Categories
    def __init__(self , n_clusters = 10 , gamma = 1.0 , random_state = None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state
        self.clustrs_marks = None
        self.names = None
        self.labels = None
    def fit(self , X , Y = None , sample_weight=None):
        self.kmeans = KMeans(self.n_clusters , random_state=self.random_state)
        self.kmeans.fit(X , sample_weight=sample_weight)
        self.clustrs_marks = self.kmeans.cluster_centers_
        self.labels = self.kmeans.labels_
        self.names = self.kmeans.fit_predict(X,sample_weight=sample_weight)
        return self
    def transform(self , X):
        return rbf_kernel(X , self.kmeans.cluster_centers_ , gamma=self.gamma)
    def get_feature_names_out(self , names = None):
        return [f'cluster{i}' for i in range(self.n_clusters)]


cluter_smi = ClusterSimilary(random_state=42)
'''
def  replace_category_to_nums_using_onehot_and_ordinal(data : pd.DataFrame , hot_or_ord : str = 'hot' , matrix_or_no : bool = True) -> np.ndarray :
    encoder = OneHotEncoder(sparse_output=matrix_or_no) , OrdinalEncoder() # here we use to most alg to convert from category to nums
    if hot_or_ord == 'hot': 
        return encoder[0].fit_transform(data) 
    elif hot_or_ord == 'ord':  # if you cat dont related use it like (bad , average , good , excllent)
        return encoder[1].fit_transform(data)
'''


cat_pipe = make_pipeline(SimpleImputer(strategy='most_frequent') , OneHotEncoder(sparse_output=False , handle_unknown='ignore'))
num_pipe = make_pipeline(SimpleImputer(strategy='mean') , StandardScaler())
cluster_pipe_line = make_pipeline(OneHotEncoder(sparse_output=False , handle_unknown='ignore') , cluter_smi)
default_process = make_pipeline(
    SimpleImputer(strategy='mean'),
    StandardScaler())


preprocessing = ColumnTransformer([
    ('cat' , cat_pipe, ['Sex' , 'Embarked']),
    ('num' , num_pipe, make_column_selector(dtype_include=np.number)),
    ('clus', cluster_pipe_line , ['Age' , 'Cabin'])] , remainder=default_process)



# analysis
'''

x_train_pre = preprocessing.fit_transform(x_train)


x_train_pre :pd.DataFrame = pd.DataFrame(x_train_pre , columns=preprocessing.get_feature_names_out())
x_train_pre['Survived'] = y_train.astype('float64') # Insert this (target) column into the data to examine the relationship between it and the data
x_train_pre['Sum_P_S'] = x_train_pre['num__SibSp'] + x_train_pre['num__Parch']
x_train_pre['Divion_S_P'] = x_train_pre['num__SibSp'] / x_train_pre['num__Parch']
x_train_pre['multi_P_S'] = x_train_pre['num__SibSp'] + x_train_pre['num__Parch']


corr_x = x_train_pre.corr()

#print(corr_x['Survived'].sort_values(ascending=False))
'''

x_train['Sum_P_S'] = x_train['SibSp'] + x_train['Parch']
x_train['Divion_S_P'] = x_train_pre['num__SibSp'] / x_train_pre['num__Parch']
x_train['multi_P_S'] = x_train_pre['num__SibSp'] + x_train_pre['num__Parch']

# train models and get the hyperparam



#cls = SGDClassifier(random_state=42) # [0.70707071 0.78451178 0.81481481]

#cls = RandomForestClassifier(random_state=42) #[0.77104377 0.82154882 0.79124579]
#param_grid = {'max_features':[5,6,7,8] , 'n_estimators':[200,100,80,60]} # [{'max_features': 6, 'n_estimators': 80}] 0.8215993974012932


#cls = SVC(random_state=42) # [0.81144781 0.83164983 0.83838384]
#param_grid = {'pre__clus__clustersimilary__n_clusters':[10,30,50 ,100],'pre__clus__clustersimilary__gamma':[0.1 ,0.05 ,0.2],'Svc__C':[1,8,5,10] ,'Svc__gamma':[0.1,0.05 , 0.2 , 0.3]} # {'C': 2, 'gamma': 0.2, 'kernel': 'rbf'} 0.830525390747599

#finally_pipe = Pipeline([('pre' , preprocessing) , ('Svc' , SVC(random_state=42 , kernel='rbf'))])

#cls = KNeighborsClassifier() # [0.78114478 0.7979798  0.80808081]
#param_grid = {'n_neighbors':[10,11,12,15],'weights':['uniform', 'distance']} # {'n_neighbors': 10, 'weights': 'uniform'} 0.809189630280585

cls = GradientBoostingClassifier(random_state=42)
param_grid = {
    'XGboost__n_estimators':[350,400,450],
    'XGboost__learning_rate':[0.1  , 0.02 , 0.03],
    'XGboost__max_depth':[3,5,8],
    'XGboost__loss':['log_loss', 'deviance', 'exponential']
}
finally_pipe = Pipeline([('pre' , preprocessing) , ('XGboost' , cls)])



grid = GridSearchCV(param_grid=param_grid , scoring='accuracy'  , cv=5 , estimator=finally_pipe)
grid.fit(x_train , y_train) 
print(grid.best_params_)
print(grid.best_score_)






'''
x_train_pre = pd.DataFrame(preprocessing.fit_transform(x_train),columns=preprocessing.get_feature_names_out())
plt.hist(x_train_pre)
plt.show()
print(pd.DataFrame(x_train_pre , columns=preprocessing.get_feature_names_out()))
cross_pred = cross_val_score(cls , preprocessing.fit_transform(x_train) ,y_train ,scoring='accuracy' , cv=10)
print(sum(cross_pred)/len(cross_pred))
'''








         
    
