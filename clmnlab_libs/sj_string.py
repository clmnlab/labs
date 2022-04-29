
# Common Libraries
import numpy as np

# Custom Libraries
import sj_sequence

# Sources

def search_string(target, search_keys, search_type = "any", exclude_keys = []):
    """
    Search string with keys in target
    
    :param target: target string(str)
    :param keys: search key(list - str)
    :param search_type: search type(str) - 'any', 'all', any is or condition, all is and condition
    :param exclude_keys: exclude key(list - str)
    
    return boolean
    """
    if search_type == "any":
        search_result = any([key in target for key in search_keys])
    else:
        search_result = all([key in target for key in search_keys])
    
    if exclude_keys != None and exclude_keys != []:
        exclude_result = not any([key in target for key in exclude_keys])
    
        return search_result and exclude_result
    else:
        return search_result

def search_stringAcrossTarget(targets, search_keys, search_type = "any", exclude_keys = []):
    """
    Search string across target strings
    
    :param target: target string(list)
    :param keys: search key(str)
    :param search_type: search type(str) - 'any', 'all', any is or condition, all is and condition
    :param exclude_keys: exclude key(list - str)
    
    return list of searched string
    """
    search_results = [search_string(target, search_keys) for target in targets]
    indexes = np.where(np.array(search_results) == True)[0]
    return sj_sequence.get_multiple_elements_in_list(targets, indexes)

def replace_all(text, dic):
    for old, new in dic.items():
        text = text.replace(old, new)
    return text

if __name__ == "__main__":
    a_string = "A string is more than its parts!"
    matches = ["more", "d"]
    
    search_string(a_string, matches, search_type = "all")
    
    replace_all("a b c", {"a" : "aa"})