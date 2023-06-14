# mse_gb, best_params_gb = grid_search_train_model(model, param_grid, X_train, y_train, X_test, y_test)
# mse_rf, best_params_rf = grid_search_train_model(RandomForestRegressor, param_grid, X_train, y_train, X_test, y_test)
# mse_xgboost, best_params_xgboost = grid_search_train_model(SGDRegressor, param_grid, X_train, y_train, X_test, y_test)
# model_gb = GridSearchCV(GradientBoostingRegressor(), param_grid)
# model_gb.fit(X_train, y_train)
# y_pred_gb = model_gb.predict(X_test)
# mse_gb = mean_squared_error(y_test, y_pred_gb)
# best_params_gb = model_gb.best_params_

# GridSearchCV with Random Forests
# model_rf = GridSearchCV(RandomForestRegressor(), param_grid)
# model_rf.fit(X_train, y_train)
# y_pred_rf = model_rf.predict(X_test)
# mse_rf = mean_squared_error(y_test, y_pred_rf)
# best_params_rf = model_rf.best_params_

# GridSearchCV with XGBoost
# model_xgboost = GridSearchCV(SGDRegressor(), param_grid)
# model_xgboost.fit(X_train, y_train)
# y_pred_xgboost = model_xgboost.predict(X_test)
# mse_xgboost = mean_squared_error(y_test, y_pred_xgboost)
# best_params_xgboost = model_xgboost.best_params_

