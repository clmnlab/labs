
# Common Libraries
from inspect import signature
import itertools
import numpy as np
import pandas as pd
import inspect
import os
import sys

# Custom Libaries
from sj_file_system import str_join

# Sources

def count_nest(x):
    """
    Count nested structure
    
    :param x: (list, tuple, set)
    
    return (int)
    """
    if isinstance(x, (list, tuple, set)):
        if len(x) == 0:
            return 1
        else:
            return 1 + count_nest(x[0])
    else:
        return 0
    
def apply_composition_function(x, functions):
    """
    data is applied to functions
    ex)
    if functions is [f,g] then x is applied as f(g(x))

    :param x: data
    :param functions: function list ex) [lambda x: x + 1, lambda x: x * 2]
    :return: data applied composition function
    """
    data_type = type(x)
    if len(functions) == 1:
        return data_type(map(functions[0], x))
    else:
        return data_type(map(functions[0], apply_composition_function(x, functions[1:])))

def recursive_map(x, function):
    """
    Each element of x is applied to function
    
    :param x: data ex) [1, [1, [2,5,3,5]], 4]
    :param function: function ex) lambda x: lambda x: round(x,1)
    :return: return the data that is applied to function recursively
    """
    if isinstance(x, (list, tuple, set)):
        t = type(x)
        return t(map(lambda e: recursive_map(e, function), x))
    else:
        return function(x)

def recursive_mapWithDepth(x, function, apply_depth):
    """
    Each element of x is applied to function
    
    :param x: data ex) [[1,2]]
    :param function: function ex) lambda x: lambda x: round(x,1)
    :param apply_depth: depth(int)
    
    :return: return the data that is applied to function recursively
    """
    if count_nest(x) == apply_depth:
        return function(x)
    else:
        if isinstance(x, (list, tuple, set)):
            t = type(x)
            return t(map(lambda e: recursive_mapWithDepth(e, function, apply_depth), x))
        
def curry(func):
    # to keep the name of the curried function:
    curry.__curried_func_name__ = func.__name__
    f_args, f_kwargs = [], {}

    def f(*args, **kwargs):
        nonlocal f_args, f_kwargs
        if args or kwargs:
            f_args += args
            f_kwargs.update(kwargs)
            return f
        else:
            result = func(*f_args, *f_kwargs)
            f_args, f_kwargs = [], {}
            return result

    return f

def apply_function(f, args):
    """
    It applies function like f(args)

    :param f: function ex) lambda x,y: x+y
    :param args: values for appling f ex) [1,1]
    
    :return: scalar, f(args)
    """
    if len(signature(f).parameters) == len(args):
        func = curry(f)
        for arg_value in args:
            func = func(arg_value)
        return func()
    else:
        raise Exception("the number of function's parameter is not matched args, len(args): ", len(args))

def flatten_2d(a_2dlist):
    """
    flatten list 2d -> 1d

    This causes dimension reduction, the dimension reduction means that square brackets is removed twice and next,
    elements are arranged and last, square braket is added 
    
    [
        [1,2,3]
    ]
    ->
    [] is removed twice, and element is arranged
    1,2,3
    
    Adding bracket
    [1,2,3]
    
    :param a_2dlist: 2d list
    :return: 1d list
    """
    return list(itertools.chain(*a_2dlist))

def flatten(l):
    try:
        return flatten(l[0]) + (flatten(l[1:]) if len(l) > 1 else []) if type(l) is list else [l]
    except IndexError:
        return []


def flatten_to_depth(data, current_depth, stop_depth):
    """
    flatten data to specific depth
    
    In stop_depth, flatten_2d applies to element.
    This causes dimension reduction, the dimension reduction means that square brackets is removed twice and next,
    elements are arranged and last, square braket is added 
    
    [
        [1,2,3]
    ]
    ->
    [] is removed twice, and element is arranged
    1,2,3
    
    Adding bracket
    [1,2,3]
    
    :param data: list
    :param current_depth: current_depth, init setting is length of data's shape
    :param stop_depth: where to apply flatten function?
    
    return flattend list
    """
    if current_depth == stop_depth:
        return data
    else:
        return flatten_2d(list(map(lambda x: flatten_to_depth(x, current_depth-1, stop_depth), data)))

def flatten_to_vector(data):
    """
    flatten from data to vector representation
    
    :param data: list
    
    return vector data array
    """
    depth = len(np.array(data).shape)
    
    if depth == 2:
        return data
    else:    
        vector_depth = 2
        dimension_reduction_depth = vector_depth + 1
    
        return flatten_to_depth(data, current_depth = len(np.array(data).shape), stop_depth = dimension_reduction_depth)

def apply_function_vector(function_vector, x_vector):
    """
    apply function_vector to x_vector
    
    :param function_vector: list of function
    :param x_vector: list
    
    return f(x)
    """
    function_index = 0
    element_index = 1
    
    def d():
        for e in zip(function_vector, x_vector):
            print(e[1])
    d()
    
    return list(map(lambda fx_set: fx_set[function_index](fx_set[element_index]), zip(function_vector, x_vector)))

class Mapping:
    """
    This class is for defining mapping function

    This class is used for namespace
    """

    @staticmethod
    def one_to_many(domain, codomain, mapping_condition):
        """
        :param domain: domain(list)
        :param codomain: codomain(list)
        :param mapping_condition: mapping condition(function - argument count is 2)

        :return: mapped list(domain to codomain, 2d list)
        """
        return list(map(lambda x: Mapping.one_value_to_many(x, list(filter(lambda y: mapping_condition(x, y), codomain))), domain))

    @staticmethod
    def one_value_to_many(x, Y):
        """

        :param x: a value(scalar)
        :param Y: codomain(list)
        :return: [[x, y1], [x, y2], ...](list)
        """
        return [[x, y] for y in Y]
    
    @staticmethod
    def space_2d(X, Y):
        """
        Using two list's combination, make 2d space
        
        :param X: list ex) [1,2,3]
        :param Y: list ex) [4,5,6]
        
        return 2d list
        """
        return list(map(lambda x: Mapping.one_value_to_many(x, Y), X))
    
    @staticmethod
    def one_to_one(X, Y, condition):
        """
        one_to_one mapping
        if one value is selected then the value is not mapped any more(successively)
        :param X: (list)
        :param Y: (list)
        :param condition: mapping condition( (x, y) -> Flag )
        
        :return: mapped values(2d list)
        """
        default_value_format = "None {0}"
        default_count = 0

        target_Y = Y[:]
        result = []
        for x in X:
            is_mapping_occurred = False
            for y in target_Y:
                if condition(x,y):
                    result.append([x, y])
                    target_Y.remove(y) # remove first occurred value y
                    is_mapping_occurred = True
                    break

            if is_mapping_occurred == False:
                defalut_value = str.format(default_value_format, str(default_count))
                result.append([x, defalut_value])
                default_count += 1
        return result

def list_map(data, function):
    """
    It executes mapping and converting to list from the mapping data

    :param data: data for mappting
    :param function: how to convert data ex) lambda x: x+3

    return: (list)
    """
    return list(map(function, data))

def map_byOrder(array, order, function, is_start = True):
    """
    mapping data considered order

    :param data: data for mappting(list)
    :param order: number of outer []
    :param function: how to convert data ex) lambda x: x+3
    :param is_start: if you call this function, needs to set is_start = True
    
    return array
    """
    if is_start == True:
        import copy
        array = copy.deepcopy(array)

    if order == 0:
        array = apply_function(function, array)
        return array

    for i in range(0, len(array)):
        array[i] = map_byOrder(array[i], order - 1, function, False)

    return array

def apply_function_vector(function_vector, x_vector):
    function_index = 0
    element_index = 1
    
    return list(map(lambda fx_set: fx_set[function_index](fx_set[element_index]), zip(function_vector, x_vector)))

def get_index_from_nested_list(data, conditions):
    """
    Returns an index that can pass by certain conditions
    
    Assume, list structure is same
    This means that the element's length of maximum depth is same
    This property can be ensured whether list is convertible to numpy or not

    :param data: list
    :param conditions: list of functions
    
    return index
    """
    assert type(np.array(data)) == np.ndarray, "Please match list's element structure"
    
    def check_function(element):
        return sum(apply_function_vector(conditions, element)) == len(conditions)

    condition_results = np.array(list(map(lambda element: True if check_function(element) else False, flatten_to_vector(data))))
    
    return condition_results.nonzero()

def relation_map(X, Y):
    """
    mapping X and Y
    
    :param X: list
    :param Y: list
    
    return relation(dict)
    """
    info = {}
    for x,y in zip(X, Y):
        if x in info.keys():
            if y not in info[x]:
                info[x] = info[x] + [y]
        else: 
            info[x] = [y]

    return info

def relation_df(keys, variables, indexes=None):
    """
    return relation using DataFrame
    must unique value for each key
    
    :param keys: list of key
    :param values: list of variables ex) [[1,2], [3,4]]
    :param indexes: name of index(list)
    
    return DataFrame
    """
    relations = []
    for variable_i in range(0, len(variables)):
        variable = variables[variable_i]
       
        print("processing: ", indexes[variable_i], "length: ", len(variable))
        
        relation = relation_map(keys, variable)

        assert len(np.unique([len(relation[key]) for key in relation.keys()])) == 1, "Not Unique Value " + indexes[variable_i]
        relations.append(pd.DataFrame(relation))

    result = pd.concat(relations)

    if indexes != None:
        result.index = indexes
    return result

def recursive_proc(x, vector_proc):
    """
    :param x: list
    :param vector_proc: procedure for vector
    """
    # TODO: np.array(x).shape로 shape 파악하는데 numpy 안쓰고 inner list가 들어있는지
    if len(np.array(x).shape) == 1:
        vector_proc(x)
    else:
        for x_e in x:
            recursive_proc(x_e, vector_proc)

def get_function_args(function_name, module_name = "__main__"):
    """
    function args
    
    :param module_name: module name(string) ex) sys.modules[__name__]
    :param function_name: function name(string) ex) clean
    
    return: arg names of function(list)
    """
    func = getattr(sys.modules[module_name], function_name)
    return func, inspect.getfullargspec(func)[0]

def multi_mapping(func_name, arg_value_pairs, module_name = "__main__"):
    """
    This function maps multiple input into output
    
    :param function_name: function name(string) ex) "clean"
    :param arg_value_pairs: arguments(list) ex) [[1, 2], [3, 4]]
    :param module_name: module name(string) ex) sys.modules[__name__]
    
    return: outpus of function(list)
    """
    func, arg_names = get_function_args(module_name = module_name, function_name = func_name)
    
    return list(map(lambda arg_value_pair: call_func_dynamically(function_name = func_name, 
                                                          argument_names = arg_names, 
                                                          arg_value_pair = arg_value_pair) ,
             arg_value_pairs))
    
def call_func_dynamically(function_name, argument_names, arg_value_pair, module_name = "__main__"):
    """
    Call function using string expression
    
    :param function_name: function name
    :argument_names: arg names(list) ex) ["a", "b"]
    :arg_value_pair: arg value pair(list) corresponding arg_name ex) [1, 2] <- number or ['1', '2'] <- string
    :param module_name: module name(string) ex) sys.modules[__name__]
    
    return: output of function
    """
    # mapping between arg name and arg value
    arg = list(map(lambda arg_name, arg_value: str_join([arg_name, arg_value], "="), argument_names, arg_value_pair))
    
    # make function call expresion
    func_call = function_name + "(" + str_join(arg, ",") + ")"
    
    # result
    result = eval(func_call, {function_name : getattr(sys.modules[module_name], function_name)})
    
    return result

####### Examples #######
if __name__ == "__main__":
    apply_function(lambda x,a,b: x+a+b, [1,2,3])

    curry(lambda x,a,b: x+a+b)(3)(2)(1)()

    recursive_map([1, [2, 3, [4],5],6], lambda x: x**2)

    apply_composition_function([1,2,3], [lambda x: x*2, lambda x: x**2])

    flatten_2d([[1,2], [3,4]])

    Mapping.one_to_many([1, 2, 3], [4, 5, 6], lambda x, y: y > x)
    Mapping.one_to_one([1, 2, 3], [4, 5, 6], lambda x, y: (x + y) > 7)

    list_map([1, 2, 3], lambda x: x + 3)

    a = np.array([[
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]])
    map_byOrder(a, 2, lambda x, y, z: [x + 10, y, z])
    
    apply_function_vector(
        function_vector = [lambda x: x+3, lambda x: x+5],
        x_vector = [1,2]
    )
    
    flatten_to_depth(
        [[[[1, 2, 3],
            [4, 5, 6]],

           [[1, 2, 3],
            [4, 5, 6]]],

         [[[1, 2, 3],
            [4, 5, 6]],

           [[1, 2, 3],
            [4, 5, 6]]]],
        len(np.array(a).shape),
        stop_depth = 3
    )
    
    flatten_to_vector([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [0, 1, 2]]])
    
    get_index_from_nested_list([[1,2,3]], [lambda x: x==1, lambda y: y==2, lambda z: z==3])

    relation_map(X=[1,2,3, 1], Y=[4,5,6, 1])
    
    recursive_proc([1,2,3], lambda v: print(len(v)))
    
    Mapping.space_2d([1,2,3], [4,5,6])
    
    def clean(a):
        print(a)
    
    get_function_args(function_name = "clean", module_name = "__main__")
    call_func_dynamically(function_name = "clean", argument_names = ["a"], arg_value_pair = ["'c'"])
    multi_mapping("clean", arg_value_pairs = [["'c'"], ["'d'"]], module_name = "__main__")
                  
    count_nest([[[1]]])
    
    recursive_mapWithDepth([[1,2]], lambda x: [x[0] + 3, x[1]], 1)
    