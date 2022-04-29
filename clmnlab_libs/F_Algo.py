# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 19:12:07 2019

@author: seojin
"""

# Common Libraries
import pandas as pd
import numpy as np

# Custom Libraries

# Sources

class DF_Helper:
    # 매개변수에 dictionary 형식으로 제한조건을 넣어주면 해당 조건에 맞는 row를 리턴해줌
    # dictionary의 key 값은 column 이름이 와야 하며 value는 해당 column에 속하는 값이 와야함
    @staticmethod
    def adjust_constraint(df, constraints):
        # constraints는 dictionary 형식이어야함
        # key는 dataframe의 column
        # value는 특정 데이터
        
        # constraints 전부를 순회하며 조건에 만족되는 row만 추출
        
        r_df = pd.DataFrame(columns = df.columns)
        
        for i in df.index:
            row = df.loc[i]
            
            insert_row = True
            for constraint_key in constraints:
                column_name = constraint_key
                constraint_value = constraints[constraint_key]
                if type(constraint_value) == list:
                    if (row[column_name] in constraint_value) == False:
                        insert_row = False
                        break
                else:
                    if row[column_name] != constraint_value:
                        insert_row = False
            
            if insert_row == True:
                r_df = r_df.append(row)
            else:
                continue
        
        r_df.index = list(range(0, len(r_df)))
        return r_df

    @staticmethod
    def group(df, col_names, agg_fun):
        # jong_ro_total_period = jong_ro.groupby([jong_ro['AMD_NM'], jong_ro['AMD_CD']]).aggregate(sum).drop('SGG_NM', axis = 1)['total']
        
        
        group_key = [df[col_name] for col_name in col_names]
        group = df.groupby(group_key).aggregate(agg_fun)
        
        # index = (1,2,3,4)
        # 모든 인덱스를 순회하고 각 인덱스를 리스트로 만들어 column으로 붙여야함
        i_d = []
        nrow = len(group)
        for i in group.index:
            if type(i) == list:
                for i_e in range(0, nrow):
                    i_d.append(i[i_e])
            else:
                i_d.append(i)
        
        index_data = np.array(np.array(i_d).reshape(nrow, len(col_names)))
        
        for i in range(0, len(col_names)):
            group[col_names[i]] = index_data[:,i]
        
        group.index = [d for d in range(0, len(group))]
        return group

# Test  #############################################################################################
if __name__=="__main__":
    import pandas as pd
    import F_Algo
    df = pd.DataFrame([[1, 2, 3], [1, 5,6], [7,8,9]],
                      columns = ['x', 'y', 'z'])
    F_Algo.DF_Helper.adjust_constraint({'x' : 1})
    F_Algo.DF_Helper.adjust_constraint(df, {'x': [1, 7], 'y': 8})

    df = pd.DataFrame([[1,1,1], [1,1,2], [1, 2, 3], [1, 2,6], [1,2,9], [1,2,3], [1,2,8]],
                      columns = ['x', 'y', 'z'])
    DF_Helper.group(df, ['x', 'y'], sum)
    
    
