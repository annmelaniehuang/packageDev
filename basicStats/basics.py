import numpy as np
import pandas as pd
from scipy.stats import trim_mean, mode
from scipy.stats import skew, skewtest, kurtosis, kurtosistest
from statistics import mean, median


DEFAULT_SIG_LEVEL = 0.05


def get_unique_values(column):
    return np.unique(column[~np.isnan(column)])


def unique_values_count(column):
    """calculate number of non-null unique values"""
    return len(get_unique_values(column))


def binary_column_recognition(column: 'np.array') -> 'boolean value':
    """Recognise columns with only two unique values for later ETL"""
    return unique_values_count(column) == 2


def categorical_column_recognition(column):
    """Recognise existing or potential categorical variables"""
    condition1 = 2 < unique_values_count(column) <= 30
    condition2 = isinstance(get_unique_values(column)[0], (np.str, np.integer))
    return condition1 and condition2


def get_mean(column):
    """Sample mean"""
    return mean(column)


def get_X_perc_trimmed_mean(column, trimmed_ratio):
    """As robust as median, trimmed mean does not completely disregard outliers
    Applicable for both normal and non-normal distributions"""
    return trim_mean(column, trimmed_ratio)


def get_median(column):
    return median(column)


def get_mode(column):
    return mode(column)


def get_min(column):
    return min(column)


def get_max(column):
    return max(column)


def get_range(column):
    return '({0:.2f} ~ {1:.2f})'.format(get_min(column), get_max(column))


def get_kurtosis(column):
    """Kirtosis measures weight of tail of data comparing to normal distribution
    higher the value, heavier the tails -> more outliers; vice versa. (Exception: uniform distribution)"""
    return kurtosis(column)


def get_kurtosis_p_values(column):
    """Valid only for sample size > 20, only p-value is printed
    null hypothesis tail weights like normal, small p-value rejects it indicates likelihood of heavy tail"""
    return kurtosistest(column)


def get_skewness(column):
    """Skewness measures lack of symmetry."""
    return skew(column)


def get_skewness_p_values(column):
    return skewtest(column)


def continuous_variable_recognition(column):
    column = get_unique_values(column) # drop nulls
    first_value = column[0]
    return unique_values_count(column) > 2 and (isinstance(first_value, (np.integer, np.float)))


def column_with_missing_value(column: 'np.array') -> 'boolean value':
    """Recognise columns with missing values
    only covers apparent cases such as
    whitespaces, nan, null, na, none, question-mark, empty parenthesis"""
    common_missing_value_format = [np.nan, ' ', '""', "''", '()', '[]', '{}', '?', '*', '.']
    for value in column:
        if value in common_missing_value_format:
            return True
        else:
            return False


class raw_data:

    def __init__(self, raw_table):
        self.raw_data = raw_table

    def get_row_number(self):
        if self.raw_data.shape[0] > 0:
            return self.raw_data.shape[0]
        else:
            print('Hey, looks like you have not read the file properly.\nTry again!')

    def get_column_number(self):
        if self.raw_data.shape[1] > 0:
            return self.raw_data.shape[1]
        else:
            print('Hey, looks like you have not read the file properly.\nTry again!')

    def sample_n_rows(self, n):
        if n < 1:
            print('Error: please enter integer >= 1')
        else:
            return self.raw_data.sample(round(n))

    def get_values(self):
        if self.get_row_number() > 0:
            return self.raw_data
        else:
            print('Hey, looks like you have not read the file properly.\nTry again!')

    def get_columns(self):
        if len(list(self.raw_data.columns.values)) > 0:
            return list(self.raw_data.columns.values)
        else:
            print('Hey, looks like you have not read the file properly.\nTry again!')

    def get_binary_column_list(self):
        binary_columns = self.raw_data.columns[self.raw_data.apply(binary_column_recognition, axis=0)]
        if len(binary_columns) > 0:
            return list(binary_columns.values)
        else:
            print('Looks like there is no binary column')

    def get_non_binary_column_list(self):
        non_binary_columns = [x for x in self.get_columns() if x not in self.get_binary_column_list()]
        if len(non_binary_columns) > 0:
            return non_binary_columns
        else:
            print('Looks like there is no non-binary column')

    def get_numeric_column_list(self):
        numeric_columns = self.raw_data.columns[self.raw_data.apply(continuous_variable_recognition, axis=0)]
        if len(numeric_columns) > 0:
            return numeric_columns
        else:
            print('Looks like there is no numerical column')

    def get_categorical_column_list(self):
        cate_columns = self.raw_data.columns[self.raw_data.apply(categorical_column_recognition, axis=0)]
        if len(cate_columns) > 0:
            return cate_columns
        else:
            print('Looks like there is no categorical column')

    def get_columns_with_apparent_missing_values(self):
        missing_value_columns = self.raw_data.columns[self.raw_data.apply(column_with_missing_value, axis=0)]
        if len(missing_value_columns) > 0:
            return list(missing_value_columns.values)
        else:
            print('Congratulations! There is no columns with obvious missing values.')

    def get_mean_traditional(self):
        traditional_mean = \
            self.raw_data.select_dtypes(exclude=[object, bool]).apply(get_mean, axis=0)
        if len(traditional_mean) > 0:
            return traditional_mean
        else:
            print('No mean values calculated, check your data ingestion.')

    def get_mean_trimmed(self, r=0.05):
        trimmed_mean = \
            self.raw_data.select_dtypes(exclude=[object, bool]).apply(lambda x: get_X_perc_trimmed_mean(x, r), axis=0)
        if len(trimmed_mean) > 0:
            return trimmed_mean
        else:
            print('No mean values calculated, check your data ingestion.')

    def get_median(self):
        medians = \
            self.raw_data.select_dtypes(exclude=[object, bool]).apply(get_median, axis=0)
        if len(medians) > 0:
            return medians
        else:
            print('No median values calculated, check your data ingestion.')

    def get_mode(self):
        modes = \
            self.raw_data.select_dtypes(exclude=[object, bool]).apply(get_mode, axis=0)
        if len(modes) > 0:
            return modes
        else:
            print('No mode values calculated, check your data ingestion.')

    def get_minimums(self):
        minimums = \
            self.raw_data.select_dtypes(exclude=[object, bool]).apply(get_min, axis=0)
        if len(minimums) > 0:
            return minimums
        else:
            print('No minimum values calculated, check your data ingestion.')

    def get_maximums(self):
        maximums = \
            self.raw_data.select_dtypes(exclude=[object, bool]).apply(get_max, axis=0)
        if len(maximums) > 0:
            return maximums
        else:
            print('No maximum values calculated, check your data ingestion.')

    def get_range(self):
        ranges = self.raw_data.select_dtypes(exclude=[object, bool]).apply(get_range, axis=0)
        if len(ranges) > 0:
            return ranges
        else:
            print('Something is wrong, check your data ingestion.')

    def get_kurtosis_report(self, sig_level=DEFAULT_SIG_LEVEL):
        kurtosis_vals = self.raw_data.select_dtypes(exclude=[object, bool]).apply(get_kurtosis, axis=0)
        test_p_vals = self.raw_data.select_dtypes(exclude=[object, bool]).\
            apply(lambda x: get_kurtosis_p_values(x)[1], axis=0)
        stat_significant = test_p_vals < sig_level
        column_names = {0: 'excess_kurtosis_values', 1: 'weight_of_tail_p_values', 2: 'is_statistically_significant'}
        if len(kurtosis_vals) > 0:
            return pd.concat([kurtosis_vals, test_p_vals, stat_significant], axis=1).rename(columns=column_names)
        else:
            print('Something is wrong, check your data ingestion.')

    def get_skewness_report(self, sig_level=DEFAULT_SIG_LEVEL):
        skewness_vals = self.raw_data.select_dtypes(exclude=[object, bool]).apply(get_skewness, axis=0)
        test_p_vals = self.raw_data.select_dtypes(exclude=[object, bool]).\
            apply(lambda x: get_skewness_p_values(x)[1], axis=0)
        stat_significant = test_p_vals < sig_level
        column_names = {0: 'skewness_values', 1: 'skewness_p_values', 2: 'is_statistically_significant'}
        if len(skewness_vals) > 0:
            return pd.concat([skewness_vals, test_p_vals, stat_significant], axis=1).rename(columns=column_names)
        else:
            print('Something is wrong, check your data ingestion.')

    def get_normality_report(self, sig_level=DEFAULT_SIG_LEVEL):
        only_continuous_variables = self.get_numeric_column_list()
        skewness_vals = \
            self.raw_data[only_continuous_variables].select_dtypes(exclude=[object, bool]).apply(get_skewness, axis=0)
        skewness_test_results = \
            self.raw_data[only_continuous_variables].select_dtypes(exclude=[object, bool]).\
            apply(lambda x: get_skewness_p_values(x)[1] < sig_level, axis=0)
        kurtosis_vals = \
            self.raw_data[only_continuous_variables].select_dtypes(exclude=[object, bool]).apply(get_kurtosis, axis=0)
        kurtosis_test_results = \
            self.raw_data[only_continuous_variables].select_dtypes(exclude=[object, bool]). \
            apply(lambda x: get_kurtosis_p_values(x)[1] < sig_level, axis=0)
        column_names = {0: 'skewness_values', 1: 'skew_is_significant',
                        2: 'excess_kurtosis', 3: 'tail_weight_is_significant'}
        if len(skewness_vals) > 0:
            return pd.concat([skewness_vals, skewness_test_results, kurtosis_vals, kurtosis_test_results],
                             axis=1).rename(columns=column_names)
        else:
            print('Something is wrong, check your data ingestion.')


