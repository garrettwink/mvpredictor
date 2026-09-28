import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd

from dataset import X_train, X_test, y_train, y_test, id_train, id_test, group_sizes

model = xgb.XGBRanker(
    objective='rank:ndcg',
    tree_method='hist'
)

model.fit(X_train, y_train, group=group_sizes)