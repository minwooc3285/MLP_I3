# model.py  
import sklearn.datasets 
import sklearn.metrics 
#from xgboost import XGBClassifier 
import sigopt 
 
# # Data preparation required to run and evaluate the sample model 
# X, y = sklearn.datasets.load_iris(return_X_y=True) 
# Xtrain, ytrain = X[:100], y[:100] 

# # Track the name of the dataset used for your Run 
# sigopt.log_dataset('iris 2/3 training, full test') 
# # Set n_estimators as the hyperparameter to explore for your Experiment 
# sigopt.params.setdefault("n_estimators", 100) 
# # Track the name of the model used for your Run 
# sigopt.log_model('xgboost') 

# # Instantiate and train your sample model 
# model = XGBClassifier( 
#   n_estimators=sigopt.params.n_estimators, 
#   use_label_encoder=False, 
#   eval_metric='logloss', 
# ) 
# model.fit(Xtrain, ytrain) 
# pred = model.predict(X) 

# # Track the metric value and metric name for each Run 
# sigopt.log_metric("accuracy", sklearn.metrics.accuracy_score(pred, y)) 



import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import sklearn.metrics

movie = pd.read_csv("./movies_metadata.csv", index_col=[0, 1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22, 23])

movie = movie.values.tolist()
del movie[19730]
del movie[29502]
del movie[35585]

max_ind = 0
for i in range(len(movie)):
    tmp = movie[i][1].split('},')
    for j in range(len(tmp)):
        if tmp[0] == '[]':
            continue
        if (max_ind < float(tmp[j].split('id')[1].split(': ')[1].split(',')[0].split('}')[0])):
            max_ind = float(tmp[j].split('id')[1].split(': ')[1].split(',')[0].split('}')[0])
        

budget = np.zeros((len(movie), 1))
popularity = np.zeros((len(movie), 1))
revenue = np.zeros((len(movie), 1))
runtime = np.zeros((len(movie), 1))

for i in range(len(movie)):
    budget[i] = float(movie[i][0])
    if np.isnan(float(movie[i][0])):
        budget[i] = 0
    
    popularity[i] = float(movie[i][2])
    if np.isnan(float(movie[i][2])):
        popularity[i] = 0
    
    revenue[i] = float(movie[i][3])
    if np.isnan(float(movie[i][3])):
        revenue[i] = 0
    
    runtime[i] = float(movie[i][4])
    if np.isnan(float(movie[i][4])):
        runtime[i] = 0

budget = (budget - budget.mean())/budget.std()
popularity = (popularity - popularity.mean())/popularity.std()
revenue = (revenue - revenue.mean())/revenue.std()
runtime = (runtime - runtime.mean())/runtime.std()

data_x = np.concatenate((budget, popularity, runtime), axis=1)
data_y = revenue

# initialization parameters
num_nodes = 128
lr = 0.001
num_iteration = 10

# Data load and split
X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, shuffle=True)

# Track the name of the dataset used for your Run 
sigopt.log_dataset('IMDB') 

# Set n_estimators as the hyperparameter to explore for your Experiment 
sigopt.params.setdefault("num_nodes", num_nodes)
sigopt.params.setdefault("learning_rate_init", lr) 
sigopt.params.setdefault("max_iter", num_iteration)  

# Track the name of the model used for your Run 
sigopt.log_model('MLP Optimization') 

# Instantiate and train your sample model 
model = MLPRegressor(hidden_layer_sizes=(num_nodes, num_nodes, num_nodes, num_nodes),learning_rate_init=sigopt.params.learning_rate_init, max_iter=sigopt.params.max_iter, verbose = False).fit(X_train, y_train)
pred = model.predict(X_test)

# Track the metric value and metric name for each Run 
sigopt.log_metric("mean_squared_error", sklearn.metrics.mean_squared_error(pred, y_test)) 
