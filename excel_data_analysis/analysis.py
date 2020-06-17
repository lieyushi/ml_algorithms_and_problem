"""
Author information:
Lieyu Shi, shilieyu91@gmail.com
"""


import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from scipy.stats import ttest_ind
from sklearn.metrics import classification_report


def read_data(filename):
    """
    This function provides a read operation and post-processing for an input csv file. It will read data from csv
    into dataframe, and besides, it will do two post-processing steps:
    1. drop the columns that all values are empty/nan/blank
    2. fill in the empty/nan/blank values with the most frequently occurred values

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


def visualize_count_by_week(df):
    """

    :param df:
    :return:
    """
    pd_week = df['date'].dt.week
    print('Week {} has maximal orders'.format(pd_week.value_counts().idxmax()))
    pd_week.plot.hist(grid=True, color='#607c8e', bins=len(pd_week.unique().tolist()))
    plt.title('Count of Orders By Week')
    plt.xlabel('Week')
    plt.ylabel('Count of Orders')
    plt.grid(axis='y', alpha=0.75)
    plt.show()


def show_mean_value_for_genders(df):
    """

    :param df:
    :return:
    """
    first = df[df['gender'] == 0]['value'].mean()
    second = df[df['gender'] == 1]['value'].mean()
    print('gender 0 has mean value: {:.2f}, gender 1 has mean value: {:.2f}'.format(first, second))

    stat, p = ttest_ind(df[df['gender'] == 0]['value'], df[df['gender'] == 1]['value'])
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distributions (fail to reject H0)')
    else:
        print('Different distributions (reject H0)')


def calculate_confusion_matrix(df):
    """

    :param df:
    :return:
    """
    results = confusion_matrix(df['gender'], df['predicted_gender'])
    print(results)
    report = classification_report(df['gender'], df['predicted_gender'])
    print(report)


if __name__ == "__main__":
    csv_data = 'screening_exercise_orders_v201810.csv'
    csv_data = read_data(csv_data)

    """
        Task A: sort the dataframe by customer_id in ascending order, and print the first 10 rows
        Answer: it assumes every row/record has customer_id
    """
    csv_data = csv_data.sort_values(by=['customer_id'])    # sort the dataframe rows by 'customer_id'
    print(csv_data.iloc[0:10])  # print the first 10 rows

    """
        Task B: plot the count of orders by week
        Answer: Week 20 has most orders
    """
    csv_data['date'] = pd.to_datetime(csv_data['date'])     # convert date from string into pd.datetime
    visualize_count_by_week(csv_data)

    """
        Task C: compute the mean value for gender 0 and 1
        Answer: Gender 0 has mean value of 363.89, while gender 1 has mean value of 350.71. Gender 0 has on average more 
        value than 1. 
        Use t-test to statistically judge whether the values of genders are statistically different or not. We use p = 
        0.05 for p-value, and the t-test result tells that null hypothesis is rejected, it means values of gender 0 and 
        gender 1 is significantly different
    """
    show_mean_value_for_genders(csv_data)

    """
        Task D: compute confusion matrix for gender and predicted_gender
        Answer: confusion matrix is 3349   3410
                                    1463   5249 
    """
    calculate_confusion_matrix(csv_data)

    """
        Task E: further discussion
        The precision for 0 is 0.70, for 1 is 0.61, and the total accuracy is 0.64, which is not high enough.
        Looks like the model tends to predict many 1 to 0, and 0 to 1. It means the model is not accurate enough. 
        Improvements:
        1. get more feature vectors. date and value purely cannot be enough to predict the gender, we might need more 
           features like, average time in shopping, payment method (cash/credit/mobil)
        2. the feature customer_id might not be good in the model. When building a ML model, can remove it
        3. Use better model, e.g., gradient boosting tree or random forest, which theoretically should have improved
           model accuracy
    """