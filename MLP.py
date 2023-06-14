import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def preprocess_data(train_df, test_df):
    X_train = torch.tensor(train_df.drop(['userId', 'movieId', 'timestamp', 'rating', 'title', 'genres', 'Directors', 'Cast'], axis=1).values, dtype=torch.float32)
    y_train = torch.tensor(train_df['rating'].values, dtype=torch.float32)
    X_test = torch.tensor(test_df.drop(['userId', 'movieId', 'timestamp', 'rating', 'title', 'genres', 'Directors', 'Cast'], axis=1).values, dtype=torch.float32)
    y_test = torch.tensor(test_df['rating'].values, dtype=torch.float32)
    return X_train, y_train, X_test, y_test

def train_model(model, X_train, y_train, learning_rate, epochs):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_train).squeeze()
        loss = nn.functional.mse_loss(y_pred, y_train)
        loss.backward()
        optimizer.step()

def evaluate_model(model, X_test, y_test):
    with torch.no_grad():
        y_pred_test = model(X_test).squeeze()
        mse = nn.functional.mse_loss(y_pred_test, y_test)
        return mse.item()

def train_and_evaluate_all_users(train_df, test_df, hidden_size1 = 64, hidden_size2 = 32, hidden_size3 = 16, learning_rate=0.01, epochs=100):
    # create an empty DataFrame
    all_predictions = pd.DataFrame(columns=['userId', 'mse'])

    for user in set(train_df['userId']):
        user_train_data = train_df[train_df['userId'] == user]
        user_test_data = test_df[test_df['userId'] == user]

        if len(user_train_data) == 0 or len(user_test_data)==0:
            continue # skip if user has no ratings

        X_train, y_train, X_test, y_test = preprocess_data(user_train_data, user_test_data)

        # Define hyperparameters
        input_size = X_train.shape[1]
        output_size = 1
        model = MLP(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)

        train_model(model, X_train, y_train, learning_rate, epochs)
        mse = evaluate_model(model, X_test, y_test)
        all_predictions = pd.concat([all_predictions, pd.DataFrame({'userId': [user], 'mse': [mse]})], ignore_index=True)
    return all_predictions

def predict_without_dimension(dim, test_df, train_df, epochs):
    test_wo_dim = test_df.drop(test_df.filter(regex=dim+'_').columns, axis=1)
    train_wo_dim = train_df.drop(train_df.filter(regex=dim+'_').columns, axis=1)
    all_predictions_wo_dim = train_and_evaluate_all_users(train_wo_dim, test_wo_dim, epochs=epochs)
    return all_predictions_wo_dim

def grid_search_best_model(model, param_grid, user_train_data, user_test_data):
    # Perform grid search
    X_train, y_train, X_test, y_test = preprocess_data(user_train_data, user_test_data)
    model = MLP(X_train.shape[1], 64, 32, 16, 1)  # Initialize the base model
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Get the best model and its hyperparameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Evaluate the best model on the test set
    y_pred = best_model(X_test).squeeze()
    mse = mean_squared_error(y_test, y_pred)

    print("Best Model:", best_model)
    print("Best Hyperparameters:", best_params)
    print("MSE on Test Set:", mse)
    return best_params