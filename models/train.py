import xgboost as xgb

from dataset import X_train, y_train, group_sizes

model = xgb.XGBRanker(
    objective="rank:pairwise",
    colsample_bytree = 0.8,
    learning_rate = 0.1,
    max_depth = 9,
    n_estimators = 50,
    subsample = 0.8
)

model.fit(X_train, y_train, group=group_sizes)