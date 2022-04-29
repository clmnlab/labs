from collections.abc import Sequence
import pandas as pd
import numpy as np
import collections
from operator import itemgetter

sample_array = np.array([
    [[-1, -1, -1],[-1, -1, -1]],
    [[1, 2, 3],[4, 5, 6]],
    [[7, 8, 9],[0, 1, 2]],
    [[4, 4, 4],[4, 4, 4]],
])

sample_3d = [
    [[1,2],[3,4],[5,6]],
    [[7,8],[9,10],[11,12]],
    [[13,14],[15,16],[17,18]],
    [[19,20],[21,22],[23,24]],
]
sample_3d = np.array(sample_3d)

def check_duplication(data):
    """
    check duplication from list
    
    :param data: data(list)
    
    return: is_duplicated
    """
    counters = collections.Counter(data)
    
    for count in counters.values():
        if count > 1:
            return True
    
    return False
    
def get_shape(lst, shape=()):
    """
    returns the shape of nested lists similarly to numpy's shape.

    :param lst: the nested list
    :param shape: the shape up to the current recursion depth
    :return: the shape including the current depth
            (finally this will be the full depth)
    """

    if not isinstance(lst, Sequence):
        # base case
        return shape

    # peek ahead and assure all lists in the next depth
    # have the same length
    if isinstance(lst[0], Sequence):
        l = len(lst[0])
        if not all(len(item) == l for item in lst):
            msg = 'not all lists have the same length'
            raise ValueError(msg)

    shape += (len(lst), )

    # recurse
    shape = get_shape(lst[0], shape)

    return shape

def make_2d_list(w, h, init_value=0):
    return [[0 for _ in range(w)] for _ in range(h)]

def is_same(array1, array2):
    """
    Checking array1 and array2 is same.
    The method to check same is to compare element by element
    
    return (boolean)
    """
    if array1.shape != array2.shape:
        return False
    
    comparison = array1 == array2
    equal_arrays = comparison.all()
    return equal_arrays

def get_multiple_elements_in_list(in_list, in_indices):
    """리스트에서 복수인덱스 값을 가져온다"""
    return [in_list[i] for i in in_indices]

def find_indexes(list, find_value, method = "equal"):
    """
    Search value from list
    
    :param list: list(list)
    :param find_value: value
    
    return: index of list matched to find_value
    """
    indexes = []
    for i in range(0, len(list)):
        if method == "equal":
            if list[i] == find_value:
                indexes.append(i)
        elif method == "in":
            if find_value in list[i]:
                indexes.append(i)
        elif method == "not_in":
            if find_value not in list[i]:
                indexes.append(i)
    return indexes

def search_dict(dictionary, keywords):
    """
    Search keywords over values in dictionary 
    
    This function iterates dictionary using keys and searchs keyword in value
    
    return: searched dictionary
    """
    keys = []
    infos = []
    for key in dictionary:
        is_not_matching = False
        for keyword in keywords:
            if keyword not in dictionary[key]:
                is_not_matching = True

        if is_not_matching == False:
            keys.append(key)
            infos.append(dictionary[key])
    return dict(zip(keys, infos))

def prob_intersection(a1, a2, threshold = False):
    """
    calculate intersection prob from array1 and array2

    :param a1: first array
    :param a2: second array
    :param threshold: cutting value about array

    return: probability
    """
    if threshold == False:
        return np.sum(a1 == a2) / a1.size
    else:
        # 두 합집합 중에서 교집합이 얼마나 되는가에 대한 확률
        return np.sum(np.logical_and(a1 > threshold, a2 > threshold)) / np.sum(np.logical_or(a1 > threshold, a2 > threshold))

def is_in(array, population):
    """
    check all element is in population

    :param array: target list
    :param population: population array

    return: True of False

    ex)
    is_in([1,2,1,2], [1,2,3]) -> True
    """
    import numpy as np
    return np.alltrue(np.array(list(map(lambda x: x in population, array))))

def get_transition(seq):
    """
    :param seq: sequence(list)
    
    return: transitions(tuples) 
    ex) 
    [('4', '1'),
     ('1', '3'),
     ('3', '2'),
     ('2', '4'),
     ('4', '3'),
     ('3', '1'),
     ('1', '2')]
    """
    transitions = []
    for unit_i in range(0, len(seq)):
        if unit_i == 0:
            continue
        else:
            previous_unit_index = unit_i - 1
            transitions.append((seq[previous_unit_index], seq[unit_i]))
    return transitions

def get_reverse_transition(seq):
    """
    :param seq: sequence(list)
    
    return: transitions(tuples) 
    ex) 
    [('4', '1'),
     ('1', '3'),
     ('3', '2'),
     ('2', '4'),
     ('4', '3'),
     ('3', '1'),
     ('1', '2')]
    """
    transitions = []
    for unit_i in range(len(seq)-1, -1, -1):
        if unit_i == len(seq) - 1:
            continue
        else:
            previous_unit_index = unit_i + 1
            transitions.append((seq[previous_unit_index], seq[unit_i]))
    return transitions

def number_of_same_transition(seq1, seq2, is_reverse=False, debug=False):
    """
    :param seq1: list
    :param seq2: list
    
    return: number of same transition
    """
    from Module import sj_datastructure

    if is_reverse == True:
        seq1_transitions = get_reverse_transition(seq1)
        seq2_transitions = get_reverse_transition(seq2)
        
    else:
        seq1_transitions = get_transition(seq1)
        seq2_transitions = get_transition(seq2)
        
    sets = sj_datastructure.Sets(seq1_transitions, seq2_transitions)
    intersection = sets.intersection()

    seq1_value_counts = dict(pd.Series(seq1_transitions).value_counts())
    seq2_value_counts = dict(pd.Series(seq2_transitions).value_counts())
    
    n_same_transition = 0
    for intersection_element in intersection:
        same_count = min(seq1_value_counts[intersection_element], seq2_value_counts[intersection_element])
        if debug == True:
            print("intersection: ", intersection_element, "count: ",same_count)
        n_same_transition += same_count
        
    return n_same_transition

def construct_layer_list(shape, init_value = 0):
    """
    construct initial layered list
    
    :param shape:
    :param init_value: initial value
    
    """
    if len(shape) == 0:
        return init_value
    else:
        result = []
        layer_iterate_count = shape[0]
        for info in range(0, layer_iterate_count):
            result.append(construct_layer_list(shape[1:], init_value))
        return result
    
def set_entry(target_list, entry_indexes, value):
    """
    set list's entry
    
    :param target_list: target list
    :param entry_indexes: index list ex) [0,0]
    :param value: value
    """
    assert len(np.array(target_list).shape) == len(entry_indexes), "list shape and indexes not matched!"
    if len(entry_indexes) == 1:
        target_list[entry_indexes[0]] = value
    else:
        set_entry(target_list[entry_indexes[0]], entry_indexes[1:], value)

def calc_2dmat_agg(mat, aggregate_func, target="all", ):
    """
    calculate 2d matrix using aggregate_func
    
    :param target: matrix calculation target / all, upper_tr, lower_tr
    :param aggregate_func: aggregate function over target entities
    """
    
    if target == "upper_tr":
        mat = np.triu(mat, 1)
    elif target == "lower_tr":
        mat = np.tril(mat, -1)

    return aggregate_func(mat.reshape(-1))

def filter_list(axis, datas, value):
    """
    Filter list
    
    :param axis: filter axis number ex) 0
    :param dats: datas(list)
    :param value: filtered by corresponding value
    
    return filtered list
    """

    return list(filter(lambda data: data[axis] == value, datas))

def filterUsingFlags(target_list, flag_list, flag_value):
    """
    Filter list using flags
    target list is filtered by using flag_list which is matched with flag_value
    
    :param target_list: target list(list)
    :param flag_list: list(element - boolean)
    :param flag_value: filter value(True or False)
    
    return target_list
    """
    return get_multiple_elements_in_list(target_list, find_indexes(flag_list, flag_value))
    
def get_itemsFromAxis(axis, datas):
    """
    Get items from specific axis over datas
    
    :param axis: axis number(int)
    :param datas: datas(list)
    
    return list
    """
    return list(map(itemgetter(axis), datas))

def interleave_array(array1, array2, interleave_count):
    """
    Interleave array2 to array1
    
    :param array1: array1(list)
    :param array2: array2(list)
    :param interleave_count: To count how many interleave
    
    return interleaved array
    """
    temp = []
    for i in np.arange(0, len(array1), interleave_count):
        for count in range(interleave_count):            
            temp.append(array1[i + count])
        for count in range(interleave_count):            
            temp.append(array2[i + count])
            
    return temp

if __name__=="__main__":
    get_transition([1,2,3])
    
    number_of_same_transition([1,2,3], [4,1,2])
    
    a = [1,2,3]
    set_entry(a, [2], 4)
    print(a)
    
    construct_layer_list((2,3))
    
    calc_2dmat_agg(np.array([
        [1,2,3],
        [4,5,6],
        [7,8,9]
    ]), sum, "upper_tr")
    
    interleave_array([1,2,3], [4,5,6], 1)
    