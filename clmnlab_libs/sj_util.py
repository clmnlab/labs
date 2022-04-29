# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 10:36:40 2020

@author: frontis
"""

import numpy as np
import cv2
import random
from Module import sj_sequence
import pandas as pd

def shift_2dmask(mask, direction, constant):
    p_mask = np.pad(mask, (1,1), mode = 'constant', constant_values= constant)
    
    if direction == 'T':
        return p_mask[2:,1:-1]
    elif direction == 'B':
        return p_mask[0:-2,1:-1]
    elif direction == 'L':
        return p_mask[1:-1,2:]
    elif direction == 'R':
        return p_mask[1:-1,0:-2]

def find_last_nonzero_row(img, is_topfirst):
    img_cp = img.copy()

    gray = cv2.cvtColor(img_cp, cv2.COLOR_RGB2GRAY)

    if is_topfirst == True:
        find_nonzero_rowIndex = gray.shape[0] - (gray!=0)[::-1,:].argmax(0)
    else:
        find_nonzero_rowIndex = gray.shape[0] - (gray!=0)[::1,:].argmax(0)
    
    find_nonzero_rowIndex = np.where(find_nonzero_rowIndex == gray.shape[0], 0, find_nonzero_rowIndex)

    return find_nonzero_rowIndex

# 1차원 등분할
def partition_d1(start_value, end_value, partition_count):
    start_x = start_value
    if partition_count <= 0:
        return []
    else:
        dx = (end_value - start_value) / partition_count

        partitions = []
        for partition_i in range(1, partition_count + 1):
            if partition_i == partition_count:
                partitions.append( (start_x, end_value) )
            else:
                partitions.append( (start_x, start_x + dx) )

            start_x += dx
        return partitions

# 1차원 등분할 버전2
def partition_d1_2(start_value, end_value, partition_count):
    return sorted(list(set(np.array(partition_d1(start_value, end_value, partition_count)).flatten())))

# return: radian, + value is clock wise from pivot to other, - value is counter clock wise from pivot to other
def angle(pivot_vector, other_vector):
    import math
    
    # https://www.edureka.co/community/32921/signed-angle-between-vectors
    x1, y1 = pivot_vector
    x2, y2 = other_vector
    
    pivot_angle = math.atan2(y1,x1)
    other_angle = math.atan2(y2,x2)
    
    return other_angle - pivot_angle

def split_value(split_value, split_count, minimum_value, maximum_value, minimum_increase):
    splits = [minimum_value] * split_count

    for _ in range(0, split_count*2):
        lottery_count = 0
        while True:
            # 증가시킬 index를 추첨

            relottery = False

            i = random.randint(0, split_count-1)
            if splits[i] + minimum_increase >= maximum_value:
                # 이미 최대값에 도달한 index는 재추첨
                relottery = True
                lottery_count += 1
            else:
                # split 값의 최대와 잔여값 중 최소 값을 증가폭의 최대로함
                remain = split_value - sum(splits)
                upper_bound = min(maximum_value - splits[i] , remain)
                if minimum_increase <= upper_bound:
                    increase = np.random.uniform(minimum_increase, upper_bound)
                    splits[i] += np.round(increase,2)

            if relottery == False or lottery_count > split_count: # 재추첨을 너무 오래 하지 않도록 세팅
                break

    non_maximum_index = []
    for i in range(0, split_count):
        if splits[i] < maximum_value:
            non_maximum_index.append(i)

    remain = split_value - sum(splits)
    if len(non_maximum_index) == 0:
        lowest_index = splits.index(min(splits))
        splits[lowest_index] += remain
    else:
        last_increasing_index = non_maximum_index[random.randint(0, len(non_maximum_index) - 1)]

        splits[last_increasing_index] += remain

    return [np.round(e, 3) for e in splits]

def get_random_sample_in_codes(sample_count, codes, appear_counts):
    """
    Getting code list from codes randomly

    :param sample_count: how many samples do you want ex) 10
    :param codes: sampling target ex) [1,2,3]
    :param appear_counts: How many appears specific value in sample_count ex) [2, 3, 5]
    :return: list of codes
    """
    result = ["empty"] * sample_count
    empty_value_indexes = sj_sequence.find_indexes(result, "empty")

    for code_i in range(0, len(codes)):
        if code_i != len(codes) - 1:
            for j in range(0, appear_counts[code_i]):
                specific_index = empty_value_indexes[random.randint(0, len(empty_value_indexes)-1)]
                result[specific_index] = codes[code_i]
                empty_value_indexes.remove(specific_index)
        else:
            for g in empty_value_indexes.copy():
                result[g] = codes[code_i]
                empty_value_indexes.remove(g)
    return result

def concat_pandas_datas(datas):
    """
    :params datas: behavior_data(pandas)

    :return: total datas(pandas)
    """

    total_data = datas[0]
    for data in datas[1:]:
        total_data = pd.concat([total_data, data], axis=0)
    return total_data

if __name__ == "__main__":
    z = split_value(split_value = 10,
                    split_count = 5,
                    minimum_value= 1,
                    maximum_value=3,
                    minimum_increase=0.5)

    get_random_sample_in_codes(10, [1,2,3], [2, 2, 6])




