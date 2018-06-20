import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble  import RandomForestRegressor
from sklearn.preprocessing import Imputer


def prepare_data(train_test_ratio = 0.8):

    train = pd.read_csv('./input/train.csv')
    train_size = int(len(train) * train_test_ratio)

    train_y = train['SalePrice'][:train_size]
    test_y = train['SalePrice'][train_size:]

    train_x = train.drop(['SalePrice'], axis=1)[:train_size]
    test_x = train.drop(['SalePrice'], axis=1)[train_size:]

    one_hot_encoded_train_predictors = pd.get_dummies(train_x)
    one_hot_encoded_test_predictors = pd.get_dummies(test_x)

    final_train_x, final_test_x = one_hot_encoded_train_predictors.align(one_hot_encoded_test_predictors,
                                                                         join='left',
                                                                         axis=1)
    imputer = Imputer()
    imputed_train_x = imputer.fit_transform(final_train_x)
    imputed_test_x = imputer.fit_transform(final_test_x)

    return imputed_train_x, train_y, imputed_test_x, test_y


def get_mae(train_x, train_y):
    return -1 * cross_val_score(RandomForestRegressor(50),
                                train_x,
                                train_y,
                                scoring='neg_mean_absolute_error').mean()

