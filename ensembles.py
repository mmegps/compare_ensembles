from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, AdaBoostRegressor

data_path = './data/diamond.csv'
diamonds = pd.read_csv(data_path)

testMode = False

if (testMode == True):
    diamonds.head(10)
    print (diamonds['cut'].unique())
    print (diamonds['color'].unique())
    print (diamonds['clarity'].unique())

diamonds = pd.concat([diamonds, pd.get_dummies(diamonds['cut'], prefix='cut', drop_first=True)],axis=1)
diamonds = pd.concat([diamonds, pd.get_dummies(diamonds['color'], prefix='color', drop_first=True)],axis=1)
diamonds = pd.concat([diamonds, pd.get_dummies(diamonds['clarity'], prefix='clarity', drop_first=True)],axis=1)
diamonds.drop(['cut', 'color', 'clarity'], axis=1, inplace=True)

target_name = 'price'
robust_scaler = RobustScaler()
X = diamonds.drop('price', axis=1)
feature_names = X.columns
X = robust_scaler.fit_transform(X)
y = diamonds[target_name]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=55)

models = pd.DataFrame(index=['train_mse', 'test_mse'],
                      columns=['KNN', 'Bagging', 'RandomForest', 'Boosting'])

#KNN Model
knn = KNeighborsRegressor(n_neighbors=20, weights='distance',
                          metric='euclidean', n_jobs=-1)
knn.fit(X_train, y_train)

#evaluation
models.loc['train_mse','KNN'] = mean_squared_error(y_pred=knn.predict(X_train),
                                                  y_true=y_train)
models.loc['test_mse','KNN'] = mean_squared_error(y_pred=knn.predict(X_test),
                                                 y_true=y_test)
print ("KNN assessment complete.")

#Bagging model
knn_for_bagging = KNeighborsRegressor(n_neighbors=20, weights='distance',
                                      metric='euclidean')
bagging = BaggingRegressor(base_estimator=knn_for_bagging, n_estimators=15,
                          max_features=0.75, random_state=55, n_jobs=-1)
bagging.fit(X_train, y_train)

#evaluation
models.loc['train_mse','Bagging'] = mean_squared_error(y_pred=bagging.predict(X_train),
                                                  y_true=y_train)
models.loc['test_mse','Bagging'] = mean_squared_error(y_pred=bagging.predict(X_test),
                                                 y_true=y_test)
print ("Bagging KNN assessment complete.")

#Random Forest model
RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55,
                          n_jobs=-1)
RF.fit(X_train, y_train)

#evaluation
models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(X_train),
                                                  y_true=y_train)
models.loc['test_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(X_test),
                                                 y_true=y_test)
print ("Random Forest assessment complete.")

#Boosting model
boosting = AdaBoostRegressor(n_estimators=50, learning_rate=0.05,
                             random_state=55)
boosting.fit(X_train, y_train)

#evaluation
models.loc['train_mse','Boosting'] = mean_squared_error(y_pred=boosting.predict(X_train),
                                                  y_true=y_train)
models.loc['test_mse','Boosting'] = mean_squared_error(y_pred=boosting.predict(X_test),
                                                 y_true=y_test)
print ("AdaBoost assessment complete.")

fig, ax = plt.subplots(figsize=(8,5))
models.loc['test_mse'].sort_values().plot(kind='barh', ax=ax)
ax.set_title('Test MSE for Regression models')
plt.savefig("MSE.png")
if (testMode == True):
    plt.show()

fig, ax = plt.subplots(figsize=(8,5))
ax.scatter(RF.predict(X_test), y_test, s=4)
ax.plot(y_test, y_test, color='red')
ax.set_title('Random Forest prediction vs observed values (test)')
ax.set_xlabel('Predicted prices')
ax.set_ylabel('Observed prices')
plt.savefig("PredictedVsObserved.png")
if (testMode == True):
    plt.show()

n_pred=10
ind_pred = RF.predict(X_test[:n_pred,])

print('Real price, Predicted price:')
for i, pred in enumerate(ind_pred):
    print (round(y_test.values[i]), round(pred), sep=', ')