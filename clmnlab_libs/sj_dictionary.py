
from operator import itemgetter

import copy

def init_dic(keys, initial_value):
    dic = {}
    for key in keys:
        dic[key] = copy.copy(initial_value)
    return dic


def find_valueInDict(info, search_value):
    """
    find value in dictionray
    :param info: (dictionary)

    return: key of dict corresponding the value
    """
    for key, value in info.items():
        if search_value in value:
            return key

    return None

def access_keys(keys, dic):
    """
    Access dictionary using keys
    
    :param keys: key of dictionary(list)
    :param dic: dictioanry
    
    return values
    """
    return itemgetter(*keys)(dic)

