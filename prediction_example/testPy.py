"""
Last name: Shi
Date: 06/11/2020
Approach: use GradientBoosting Tree can achieve highest AUC (0.849), than LogisticRegression (0.557) and
RandomForest (0.818). Provide robust function API for data laoding, processing and cleaning

Comment: I don't think this task can be finished within one hour. ML algorithm is never a big problem, but the data
cleaning and processing part is. For the nan or empty values, previously I just want to remove any rows that have
nan or empty, but as long as the requirement is to consider all cases, so I assign default values to the blank space,
with the most frequently appearing elements. Also, how to build readable and integrated function API is also something
that takes time to finish.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from pandas.api.types import is_string_dtype
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report


def read_data(filename):
    """

    :param filename: the path of a file
    :return: return a dataframe object
    """
    df = pd.read_csv(filename)
    df = df.dropna(how='all')   # drop the column that all rows are empty, e.g., the 'tenure' in test.csv
    print(df.columns.values.tolist())
    filled = {}     # for those column that has empty, should be assigned to the most frequent elements
    for feature in df.columns:
        if df[feature].isnull().values.any():
            unique_count = df[feature].value_counts(dropna=True)
            unique_names = unique_count.index.values.tolist()
            unique_count = unique_count.values.tolist()
            most_frequent = None
            maximal = -1
            for i in range(len(unique_count)):
                if unique_count[i]>maximal:
                    maximal = unique_count[i]
                    most_frequent = unique_names[i]
            filled[feature] = most_frequent
    df = df.fillna(filled)
    return df


def visualize_features(df):
    """

    :param df: an input dataframe object
    :return: visualize the data
    """
    [row, col] = df.shape
    print('there are {} rows and {} columns'.format(row, col))
    column_names = df.columns.tolist()
    fig, axs = plt.subplots(2, int(col/2))
    for i in range(2):
        for j in range(int(col/2)):
            feature = column_names[5*i+j]
            # unique_values = df[feature].value_counts()
            # x = unique_values.index.values.tolist()
            # y = unique_values.values.tolist()
            axs[i, j].hist(df[feature])
            axs[i, j].set_title('Frequency of {}'.format(feature))
            axs[i, j].set(xlabel=feature, ylabel='frequency')
    plt.show()


def select_features(df):
    """
    select features based on correlation. If two features have correlation larger than 0.9, will just keep one
    and delete the other. The feature selection is referenced at
    https://towardsdatascience.com/feature-selection-correlation-and-p-value-da8921bfb3cf

    :param selected_columns: an input for selected features
    :param df: a dataframe object
    :return: a selected feature list and the features itself
    """
    print('selecting features based on correlations')
    [row, col] = df.shape
    column_names = df.columns.tolist()

    data = df[column_names[0: (col - 1)]]
    corr = data.corr()
    columns = np.full((corr.shape[0],), True, dtype=bool)
    deleted = []
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if corr.iloc[i, j] >= 0.9:
                if columns[j]:
                    columns[j] = False
    selected_columns = data.columns[columns]
    x = df[selected_columns]
    y = df[column_names[col - 1]]
    return selected_columns, x, y


def convert_text_to_numeric(df):
    """

    :param df: a dataframe object
    :return: return an dataframe object
    """
    column_names = df.columns.tolist()
    for feature in column_names:
        if is_string_dtype(df[feature]):
            print('Covert text data {} into numeric'.format(feature))
            le = preprocessing.LabelEncoder()
            print(df[feature].unique())
            le.fit(df[feature].unique())
            df[feature] = le.transform(df[feature])
    return df


def model_creation(x_, y_, method_option):
    """

    :param x_train: features of training data
    :param y_train: labels of training data
    :param x_test: features of testing data
    :param y_test: labels of testing data
    :return:
    """
    x_train, x_test, y_train, y_test = train_test_split(x_, y_, test_size=0.33, random_state=0)
    if method_option=='LogisticRegression':
        model = LogisticRegressionCV(cv=5)
    elif method_option=='RandomForest':
        model = RandomForestClassifier(n_estimators=300, max_depth=100, max_features='sqrt')
    elif method_option=='GradientBoosting':
        model = GradientBoostingClassifier(random_state=0)
    model.fit(x_train, y_train)

    y_predicted = model.predict(x_test)
    scores = cross_val_score(model, x_train, y_train, cv=5, scoring='roc_auc')
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_predicted))
    print('\n')
    print("=== Classification Report ===")
    print(classification_report(y_test, y_predicted))
    print('\n')
    print("=== All AUC Scores ===")
    print(scores)
    print('\n')
    print("=== Mean AUC Score ===")
    print("Mean AUC Score - Random Forest: ", scores.mean())
    return model


def predict_label(model, test_data, threshold):
    """

    :param model: a prediction model
    :param test_data: test data to predict the labels
    :return:
    """
    print('Predicted outcome for test.csv data:')
    labels = model.predict_proba(test_data)
    test_data['predicted_outcome'] = (labels[:,1] > threshold).astype(int)
    print(test_data.iloc[0:5])


if __name__ == "__main__":
    train_data = 'data/train.csv'
    train_data = read_data(train_data)
    # visualize_features(train_data)
    train_data = convert_text_to_numeric(train_data)
    selected, x_, y_ = select_features(train_data)
    del train_data

    option = ['RandomForest', 'LogisticRegression', 'GradientBoosting']
    ml_model = model_creation(x_, y_, option[1])

    test_data = 'data/test.csv'
    test_data = read_data(test_data)
    test_data = convert_text_to_numeric(test_data)
    threshold = 0.8
    predict_label(ml_model, test_data, threshold)

