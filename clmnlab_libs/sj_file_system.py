
import time
from ast import literal_eval
from os import stat, path
import os
import csv
import pandas as pd
import pickle 

def str_join(strs, deliminator = "_"):
    """
    join string

    :param strs: list of string
    :param deliminator: deliminator

    return: combination string
    """
    strs = list(filter(lambda str: str != "", strs))
    
    if len(strs) == 1:
        return str(strs[0])
    else:
        return strs[0] + deliminator + str_join(strs[1:], deliminator)

def str_join_multi_deliminators(strs, deliminators = ["_"]):
    """
    join string

    :param strs: list of string
    :param deliminator: deliminators

    return: combination string
    """
    strs = list(filter(lambda str: str != "", strs))
    
    if len(strs) == 1:
        return str(strs[0])
    else:
        current_deliminator = deliminators[0]
        next_deliminators = deliminators[1:] + [current_deliminator]
        return strs[0] + deliminators[0] + str_join_multi_deliminators(strs[1:], next_deliminators)
    
def file_name(path):
    """
    :param path: file_path
    :return: string(file name)
    """
    from sys import platform

    if platform == 'win32':
        deliminator = '\\'
    else:
        deliminator = '/'

    return path.split(deliminator)[-1]

def wait_for_write_finish_download(file_path, wait_seconds, exception):
    """
    process is waited until file is not updated

    :param file_path: file path
    :param wait_seconds: waiting timeout seconds if wait time is elapsed until the seconds then raise exception ex) 5
    :param exception: timeout exception ex) Exception("Timeout")
    """
    # it determines download completed if the file's size is same before and after per 1 second
    count = 0
    last_size, size= -1, 0
    while size != last_size:
        time.sleep(1)
        count += 1
        last_size, size = size, stat(file_path).st_size
        if count >= wait_seconds:
            raise exception

def wait_for_file_download(file_info_path, file_path, wait_seconds, exception):
    """
    process is waited until file is fully downloaded

    :param file_info_path: file path containing file size
    :param file_path: need to wait downloading file path
    :param wait_seconds: waiting timeout seconds if wait time is elapsed until the seconds then raise exception ex) 5
    :param exception: timeout exception ex) Exception("Timeout")
    """
    count = 0
    while compare_file(file_info_path, file_path) != True:
        time.sleep(1)
        count += 1
        if count >= wait_seconds:
            raise exception
    return count

def read_file_size(file_info_path, file_name):
    """
    This function reads file size of file_name in the file located file_info_path

    :param file_info_path: file path containing file size
    :param file_name: file_name ex) asahi.jpg
    :return: integer(bytes of file_size)
    """
    with open(file_info_path, "r") as f:
        str_file_info = f.read()
        file_info = literal_eval(str_file_info)
        try:
            size = file_info[file_name]
        except:
            return -1
        return size

def compare_file(file_info_path, file_path):
    """
    This function compares file size between file_path and file_info_path

    :param file_info_path: file path containing file size
    :param file_path: file path for checking
    :return:
    """
    file_info_size = read_file_size(file_info_path, file_name(file_path))
    if file_info_size == -1:
        return True # if file_info is not existed, return true for convenience
    else:
        file_current_size = stat(file_path).st_size
        return file_info_size == file_current_size

class CsvManager:
    def __init__(self, dir_path="", file_name=""):
        self.dir_path = dir_path
        self.file_name = file_name
        self.file_path = os.path.join(self.dir_path, self.file_name + ".csv")

    def read_csv_from_pandas(self):
        return pd.read_csv(self.file_path)

    def read_csv(self):
        try:
            with open(self.file_path, 'r', newline='') as f:
                reader = csv.reader(f)
                return [line for line in reader]
        except FileNotFoundError as e:
            return []

    def write_header(self, headers):
        if len(self.read_csv()) == 0:
            with open(self.file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
        else:
            print("There exists header already")

    def write_row(self, row):
        with open(self.file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

def save(obj, file_path):
    """
    Save file using pickle
    
    :param obj: object to save
    :param file_path: save file path
    """
    filehandler = open(file_path, 'wb') 
    pickle.dump(obj, filehandler)
   
def load(file_path):
    """
    Load file using pickle
    
    :param file_path: load file path
    """
    filehandler = open(file_path, 'rb') 
    return pickle.load(filehandler)

def exist_all_files(file_paths):
    """
    Check all file is exists
    
    :param file_paths: file paths(list, element-string)
    
    return (boolean)
    """
    return sum([os.path.exists(file_path) for file_path in file_paths]) == len(file_paths)

def rename(dir_path, target_str, replace_str):
    """
    Rename file name
    
    :param dir_path: directory path 
    :param target_str: target file name under dir_path
    :param target_str: replace file name under dir_path
    """
    target = os.path.join(dir_path, target_str)
    replace = os.path.join(dir_path, replace_str)
        
    command = str_join(["mv", target, replace], deliminator = " ")
    print(command)
    os.system(command)
    
if __name__ == "__main__":
    """
    Example of CsvManager 
    """
    csv_m = CsvManager(dir_path="/Users/yoonseojin/Downloads",
                       file_name="test4")

    csv_m.write_header(["a","b","cdefgh"])
    csv_m.write_row(["얍", "얍얍", "얍얍얍"])

    sj_file_system.save(pd.DataFrame({"A:1"}), "./dd")
    
    sj_file_system.load("./dd")
    
    sj_file_system.rename("/sdafsd", "abc", "def")
