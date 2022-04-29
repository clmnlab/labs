
from enum import Enum
from Module import sj_higher_function
from Module.sj_higher_function import flatten
import numpy as np
import pandas as pd

class Aggregate(Enum):
    mean = 1 << 0
    median = 1 << 1

def flags(list, condition):
    """
    return data of flags

    :param list: data
    :param condition: check data function
    :return: flags of data
    """
    return [True if condition(e) else False for e in list]

def choice(list1, list2, flags):
    """
    if flag is True then select list1's component
    otherwise select list2's component

    :param list1: first set of data
    :param list2: second set of data
    :param flags: criteria of choice algorithm
    :return:
    """
    if len(list1) == len(list2) == len(flags):
        result = []
        for i in range(0, len(flags)):
            if flags[i] == True:
                result.append(list1[i])
            else:
                result.append(list2[i])
        return result
    else:
        raise Exception("length is not compatible among list1, list2 and flags")

def counter(data, bin_ranges, equal_direction = "None"):
    """
    It counts number of data according to a range of bin
    if equal_direction is Left, equal sign of inequality is attached at lower bound
        so it means that if we check 3 and equal_direction is Left, we check the condition like lower_bound <= 3 < upperbound
        if equal_direction is None, we check the condition like lower_bound < 3 < upperbound

    :param data: list ex) [1,2,3,4]
    :param bin_ranges: range of bin ex) [(0,1), (1,2), (2,3)]
    :param equal_direction: direction of equal(None, Left, Right) ex) "Left"
    :return: number of data ex) { (0,1) : 3, (1,2) : 2 }
    """
    method_left_equal = lambda x, lower_bound, upper_bound: True if (x >= lower_bound and x < upper_bound) else False
    method_right_equal = lambda x, lower_bound, upper_bound: True if (x > lower_bound and x <= upper_bound) else False
    method_none_equal = lambda x, lower_bound, upper_bound: True if (x > lower_bound and x < upper_bound) else False


    if equal_direction == "None":
        selected_method = method_none_equal
    elif equal_direction == "Left":
        selected_method = method_left_equal
    elif equal_direction == "Right":
        selected_method = method_right_equal

    result = {}
    for key in bin_ranges:
        result[key] = 0

    for e in data:
        for range in bin_ranges:
            lower_b = range[0]
            upper_b = range[1]
            if selected_method(e, lower_b, upper_b) == True:
                result[range] = result[range] + 1
                break
    return result

def df_to_series(data):
    return list(map(lambda x: x[1], list(data.iterrows())))

def search_column_index(data, keywords, search_mode = 1):
    """
    search column_index from passed condition
    
    :param data: dataframe
    :param keywords: search keywords, if multiple index the keywords are list
    :param search_mode: 1 <-> correct, 0 <-> contain
    return index array
    """
    def search_method(keyword):
        if keyword == None or keyword == "":
            return lambda element: True
        else:
            if search_mode == 1:
                return lambda element: keyword == element
            else:
                return lambda element: keyword in element
    
    conditions = [search_method(keyword) for keyword in keywords]
    
    return sj_higher_function.get_index_from_nested_list(data, conditions)[0]

def get_column_keywords(data, keywords, search_mode = 1):
    """
    search data from passed condition
    
    :param data: dataframe
    :param keywords: search keywords, if multiple index the keywords are list
    :param search_mode: 1 <-> correct, 0 <-> contain
    
    # get_column_keywords(grade_data_1_2, ["Cognitive", "습", None, None])
    
    return dataframe
    """
    columns = list(data.columns)
    column_indexes = search_column_index(data=data, keywords=keywords, search_mode=search_mode)

    result = data.iloc[:, column_indexes]
    
    return result

def get_columns_keywords(data, keywords_list, mode=0):
    col_data_list = []
    for keywords in keywords_list:
        col_data = get_column_keywords(data, keywords=keywords, search_mode=mode)
        col_data_list.append(col_data)
    
    return pd.concat(col_data_list, axis=1)
    
def search_multi_conditions(data, search_columns, filter_funcs):
    """
    filter data using filter funcs over search_columns
    
    :param search_columns: list of column name ex [ ["학생코드", "", "", ""], ...] <- if use multindex column
    :filter_funcs: Apply filter function over the column vector
    
    return DataFrame
    """
    
    conditions = []
    for search_column, filter_func in zip(search_columns, filter_funcs):
        column_data = to_series(get_column_keywords(data, search_column, search_mode=1))
        condition = column_data.apply(filter_func)
        
        conditions.append(condition)
        
    condition_matrix = pd.concat(conditions, axis = 1)
    final_condition = condition_matrix.apply(lambda x: sum(x), axis=1) == len(search_columns)

    return data.iloc[list(final_condition),:]

def reverse_dict(dictionary):
    new_dict = { dictionary[k]:k for k in dictionary}
    return new_dict

def is_nan(x):
    if type(x) == str:
        return x == "nan"
    else:
        return np.isnan(x)

def is_not_nan(x):
    return is_nan(x) == False

def to_series(one_column_df):
    return one_column_df.iloc[:,0]

def search(data,
           search_columns,
           filter_funcs,
           mode=1,
           showing_columns = None,):
    searched = search_multi_conditions(data=data, 
                       search_columns=search_columns,
                        filter_funcs=filter_funcs
                       )
    if showing_columns == None:
        return searched
    else:
        return get_columns_keywords(searched, keywords_list=showing_columns, mode=mode)

def group(data, by, group_function, filter_func=None):
    """
    grouping data
    
    :param by: grouping column_name
    :param group_function: function to apply group data
    
    return DataFrame
    """
    
    grouping_targets = []
    group_func_result = []
    for grouping_target, group_data in data.groupby(by):
        grouping_targets.append(grouping_target)
        group_func_result.append(group_function(group_data))

    result = pd.DataFrame([group_func_result])
    result = result.T
    result.index = grouping_targets
    
    if filter_func != None:
        result = result[list(map(filter_func, group_func_result))]
    
    return result

def compare_df(df1, df2, ignore_index=True):
    if ignore_index:
        df1_ = df1.reset_index(drop=True)
        df2_ = df2.reset_index(drop=True)
    else:
        df1_ = df1
        df2_ = df2
    
    return df1_.compare(df2_, 
                align_axis=1, 
                keep_equal=True, 
                keep_shape=True)

def change_df(data, column_name, apply_func):
    """
    change data
    
    :param data: DataFrame
    :param column_name: name of column. if you use MultiIndex, column_name is list
    :param apply_func: function to apply column data
    
    return DataFrame
    """
    data_ = data.copy()
    
    target_column_index = search_column_index(data, column_name)
    assert len(target_column_index) == 1, "searched multiple columns!!"

    applied_data = to_series(data_.iloc[:, target_column_index]).apply(apply_func)

    data_.iloc[:, target_column_index] = pd.DataFrame(applied_data)
    return data_

def replace_str(entry, replace_targets, replace_results):
    """
    :param entry: data
    :param replace_targets: replace target string
    :param replace_results: target <-> string
    """
    assert type(entry) == str, "Please input string to entry"
    assert len(replace_targets) == len(replace_results), "Please match length between replace_targets, replace_results"
    
    if len(replace_targets) == 0:
        return entry
    else:
        replace_result = entry.replace(replace_targets[0], replace_results[0])
        return replace_str(replace_result, replace_targets[1:], replace_results[1:])


def one_hot_encoding(string, split_str, variables):
    """
    return one-hot encoding 
    
    :param string: string
    :param split_str: split string
    :param variables: one_hot_encoding variables
    
    return: one-hot encoding(DataFrame)
    """
    result = np.zeros(len(variables))
    
    targets = string.split(split_str)
    for target in targets:
        for variable_i in range(0, len(variables)):
            if target in variables[variable_i] and len(target)>0:
                result[variable_i] = 1

    result_df = pd.DataFrame(np.array(result, dtype=int)).T
    result_df.columns = variables
    return result_df
            
def one_hot_encodings(strings, split_str, variables):
    """
    return one-hot encoding 
    
    :param string: string
    :param split_str: split string
    :param variables: one_hot_encoding variables
    
    return: one-hot encoding(DataFrame)
    """
    return pd.concat(list(map(lambda string: one_hot_encoding(string, split_str, variables),
                              strings)), 
                     axis=0)

def remove_non_need_column(data, non_need_keywords):
    """
    remove non need column
    
    :param data: dataframe
    :param non_need_keywords: search keyword for column
    
    return: DataFrame
    """
    search_non_need_column_indexes = []
    for keywords in non_need_keywords:
        search_non_need_column_indexes.append(list(search_column_index(data,
                                                                       keywords = keywords)))
        search_non_need_column_indexes = flatten(search_non_need_column_indexes)
    
    # Extract need index
    need_indexes = []
    col_indexes = np.arange(0, len(data.columns))
    for col_index in col_indexes:
        if col_index not in search_non_need_column_indexes:
            need_indexes.append(col_index)

    return data.iloc[:, need_indexes]

def sort_columns(df, sort_column_array):
    access_first_entry = 0

    column_index_info = {}
    for column_i in range(0, len(df.columns)):
        column = df.columns[column_i]
        column_index_info[column] = column_i
    
    return df.iloc[:, list(map(lambda column: column_index_info[column], sort_column_array))]

####### Examples #######
if __name__ == "__main__":
    is_nan(np.NaN)
    is_not_nan(np.NaN)

    search_column_index(pd.DataFrame({
                            "A" : [1,2,3],
                            "B": [4,5,6]
                        }), 
                    ["B"],
                    search_mode=1)

    get_column_keywords(pd.DataFrame({
                                "A" : [1,2,3],
                                "B": [4,5,6]
                            }),
                          keywords= ["A"])

    search_multi_conditions(data=pd.DataFrame({
                            "A" : [1,2,3],
                            "B": [4,5,6]
                        }), 
                        search_columns=[["A"]],
                        filter_funcs=[lambda x: x>=2]
                       )

    get_columns_keywords(pd.DataFrame({
        "A" : [1,2,3],
        "B" : [4,5,6],
        "C": [5,5,5]
    }), keywords_list = [
                ["A"],
                ["B"]
            ])

    search(pd.DataFrame({
        "A": [1,2,3],
        "B": [4,5,6],
        "C": [7,8,9]}
        ),
           search_columns = ["A"],
           filter_funcs=[lambda x: x==1],
           showing_columns = [["A"], ["B"]],
           mode=0
          )

    group(pd.DataFrame({
        "A" : [1,2,3,1],
        "B": [4,5,6,1]}),
        by = "A",
        group_function=lambda x: sum(x["B"]),
        filter_func=lambda x: x > 5
    )

    compare_df(pd.DataFrame({
                                "A" : [1,2,3],
                                "B": [4,5,6]
                            }), 
        pd.DataFrame({
                                "A" : [2,3,4],
                                "B": [4,5,6]
                            }))

    change_df(pd.DataFrame({
                                "A" : [1,2,3],
                                "B": [4,5,6]
                            }),
             column_name="A",
             apply_func = lambda x: x+1)

    replace_str("abc", ["a","b"], ["d", "f"])

    one_hot_encoding("a b c", " ", ["a", "b", "d"])
    one_hot_encodings(["a b c", "b"], " ", ["a", "b", "d"])

    remove_non_need_column(pd.DataFrame({
                                    "A" : [1,2,3],
                                    "B": [4,5,6]
                                }),
                          non_need_keywords=["A"])
    
    sort_columns(pd.DataFrame({
                            "A" : [1,2,3],
                            "B": [4,5,6]
                        }),
            ["B","A"])
