from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre processes data for modeling. Receives train, val and test dataframes
    and returns numpy ndarrays of cleaned up dataframes with feature engineering
    already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarrary
        val : np.ndarrary
        test : np.ndarrary
    """

    # Print shape of input data
    print("Input train data shape: ", train_df.shape)
    print("Input val data shape: ", val_df.shape)
    print("Input test data shape: ", test_df.shape, "\n")

    # Make a copy of the dataframes
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    # 1. Correct outliers/anomalous values in numerical

    # columns (`DAYS_EMPLOYED` column).
    working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

    # Fit and transform ordinal columns
    ordinal_encoder = OrdinalEncoder()
    ordinal_columns = ["EMERGENCYSTATE_MODE", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_CONTRACT_TYPE"]

    working_train_df[ordinal_columns] = ordinal_encoder.fit_transform(working_train_df[ordinal_columns])
    working_val_df[ordinal_columns] = ordinal_encoder.transform(working_val_df[ordinal_columns])
    working_test_df[ordinal_columns] = ordinal_encoder.transform(working_test_df[ordinal_columns])

    # Fit and transform one-hot columns
    one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    one_hot_columns = ["CODE_GENDER", "NAME_TYPE_SUITE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
                    "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE", "WEEKDAY_APPR_PROCESS_START",
                    "FONDKAPREMONT_MODE", "HOUSETYPE_MODE", "WALLSMATERIAL_MODE"]

    # Fit and transform on training data
    transformed_train_data = one_hot_encoder.fit_transform(working_train_df[one_hot_columns])
    train_columns = one_hot_encoder.get_feature_names_out(one_hot_columns)
    working_train_df[train_columns] = transformed_train_data

    # Transform on validation and test data
    transformed_val_data = one_hot_encoder.transform(working_val_df[one_hot_columns])
    working_val_df[train_columns] = transformed_val_data

    transformed_test_data = one_hot_encoder.transform(working_test_df[one_hot_columns])
    working_test_df[train_columns] = transformed_test_data

    # Drop one-hot columns
    working_train_df.drop(one_hot_columns, axis=1, inplace=True)
    working_val_df.drop(one_hot_columns, axis=1, inplace=True)
    working_test_df.drop(one_hot_columns, axis=1, inplace=True)

    # Fit and transform on training data
    numerical_columns = working_train_df.select_dtypes(include=np.number).columns

    imputer = SimpleImputer(strategy="median")
    imputer.fit(working_train_df[numerical_columns])

    working_train_df[numerical_columns] = imputer.transform(working_train_df[numerical_columns])
    working_val_df[numerical_columns] = imputer.transform(working_val_df[numerical_columns])
    working_test_df[numerical_columns] = imputer.transform(working_test_df[numerical_columns])

    # Fit and transform on training data
    numerical_columns = working_train_df.select_dtypes(include=np.number).columns

    scaler = MinMaxScaler()
    scaler.fit(working_train_df[numerical_columns])
    working_train_df[numerical_columns] = scaler.transform(working_train_df[numerical_columns])
    working_val_df[numerical_columns] = scaler.transform(working_val_df[numerical_columns])
    working_test_df[numerical_columns] = scaler.transform(working_test_df[numerical_columns])

    # Convert to numpy arrays
    train = np.array(working_train_df)
    val = np.array(working_val_df)
    test = np.array(working_test_df)

    return train, val, test
