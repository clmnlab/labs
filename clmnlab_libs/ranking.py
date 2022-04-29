# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 20:23:39 2019

@author: seojin
"""

# Common Libraries
import pandas as pd

# Sources

def top_ranking_df(df, column_name, rank_range):
    sorted_df = df.sort_values(by=column_name, ascending = False)
    
    ranked_s = sorted_df[column_name].rank(ascending = False)
    r_i = []
    for index in ranked_s.index:
        if ranked_s[index] >= rank_range[0] and ranked_s[index] <= rank_range[1]:
            r_i.append(index)
    
    return sorted_df.loc[r_i]

if __name__=="__main__":
    import ranking
    df1 = pd.DataFrame(
        [[1,3,48], [2,2,50], [3,1,49]], index = [3,5,6]
        )
    top_ranking_df(df1, 1, [1,3])



