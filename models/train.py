import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd

from dataset import X_train, X_test, y_train, y_test, id_train, id_test

model = xgb.XGBRegressor(
    colsample_bytree = 0.8,
    learning_rate = 0.1,
    max_depth = 9,
    n_estimators = 50,
    subsample = 0.8
)

model.fit(X_train, y_train)