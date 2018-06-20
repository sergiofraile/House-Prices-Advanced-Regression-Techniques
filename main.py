import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
# from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer
from utils import *

# Hyperparameters
train_test_ratio = 0.8

# Prepare data
train_x, train_y, test_x, test_y = prepare_data(train_test_ratio)


mae_one_hot_encoded = get_mae(train_x, train_y)

print('Mean Absolute Error with One-Hot Encoding: ' + str(int(mae_one_hot_encoded)))

