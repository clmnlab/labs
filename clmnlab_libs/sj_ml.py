# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 11:04:18 2020

@author: seojin
"""

# Common Libraries
import numpy as np
import time
from tensorboardX import SummaryWriter
from sklearn.utils import shuffle
import torch
import pandas as pd
import os

# Custom Libraries
import sj_higher_function
import sj_file_system
from sj_file_system import str_join

# Sources

def numerical_gradient(f, x):
    """
    find numerical gradient about f at x

    :param f: function ex) lambda x,a,b: x**2 + a**3 + b
    :param x: specific value ex) [2,3,4]
    :return: list(numerical gradient)
    """
    x = np.array(x, dtype = np.float64)
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = sj_higher_function.apply_function(f, x)

        x[idx] = tmp_val - h
        fxh2 = sj_higher_function.apply_function(f, x)
        
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
        
    return grad

def gradient_descent(f, init_x, lr, step_num = 100):
    """
    apply simultaneous gradient descent

    :param f: function for fitting ex) lambda x: x**2
    :param init_x: initial value ex) [2]
    :param lr: learning rate ex) 0.001
    :param step_num: number of learning steps ex) 100
    :return: fitted value
    """
    x = init_x 
    # Loss에 x를 input으로 넣을때, gradient를 구함
    # 파라미터 업데이트
    for i in range(0, step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
        
    return x

def one_vs_all(X_train, Y_train, X_test, Y_test):
    """
    train one vs all model and predict

    :params X: datas
    :params y: labels

    :return: predicts, tests, labels
    """

    from sklearn.svm import SVC
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import Pipeline

    svc_ova = OneVsRestClassifier(Pipeline([
        ('svc', SVC(kernel='linear'))
    ]))

    svc_ova.fit(X_train, Y_train)
    y_pred_ova = svc_ova.predict(X_test)

    label_kinds = np.unique(Y_test)

    return y_pred_ova, Y_test, label_kinds


def plot_recall(tests, preds, targets, colors):
    """
    :params test: test data
    :params preds: predict data
    :params targets: label's class ex) ["1-4-2-3-1", "3-2-1-4-5"]
    :params colors: class's color
    """
    import matplotlib.pylab as plt
    from sklearn.metrics import classification_report

    recalls = []
    report = classification_report(tests, preds, output_dict=True)

    for target in targets:
        recalls.append(report[target]["recall"])

    bar_width = 0.35
    alpha = 0.5

    import matplotlib.patches as mpatches
    for i in range(0, len(recalls)):
        plt.bar(i + 1 + bar_width,
                recalls[i],
                bar_width,
                color=colors[i],
                alpha=alpha)

        plt.ylabel("recall (prob)")

    patchs = []
    for i in range(0, len(recalls)):
        patchs.append(mpatches.Patch(color=colors[i], label=targets[i], alpha=0.5))
        plt.legend(handles=patchs, loc="upper center")

class SaveModel:
    def __init__(self, model, name, params):
        self.model = model
        self.name = name
        self.params = params

class ExecModel:
    def __init__(self, 
                 model, 
                 name, 
                 log_dir_path, 
                 prefix="",
                 is_shuffle=True):
        """
        Wrapper of Model
        This class logs accuracy per training
        
        :param model: model
        :param name: model name
        :param log_dir_path: tensorboard log directory path
        :param prefix: prefix of log and file(for saving model)
        :param is_shuffle: data is shuffled when model is trained
        """

        self.name = name
        print("Model Name: ", name)
        
        self.model = model
        self.is_shuffle = is_shuffle
        print("is_shuffle", is_shuffle)
        
        # log
        self.log_dir_path = log_dir_path
        self.logger = SummaryWriter(log_dir=log_dir_path)
        self.prefix = prefix
        self.log_group = str_join([self.prefix, name])
    
        self.train_evals = {}
        self.train_evals["ep_train_eval"] = []
        self.train_evals["batch_train_eval"] = []
        
        self.test_evals = {}
        self.test_evals["ep_test_eval"] = []
        self.test_evals["batch_test_eval"] = []
        
        self.save_file_name = str_join([self.prefix, name])
        
        self.auxiliary_criterions = []
        
        # save
        self.save_proc = None
        
    def append_auxiliary_criterion(self, criterion, criterion_name):
        """
        Append criterion
        
        :param criterion: criterion function ex) lambda preds, ys, : sum(ys - preds)
        """
        self.auxiliary_criterions += [(criterion_name, criterion)]
        
        self.train_evals[str_join(["ep", "train", criterion_name])] = []
        self.test_evals[str_join(["ep", "test", criterion_name])] = []
    
    def set_pf_params(self, epoch, batch_size):
        if self.model.is_partial_fit:
            print("this model is fitted by partial!!")
            
            self.epoch = epoch
            self.batch_size = batch_size
            
            print("epoch: ", self.epoch, "batch_size", self.batch_size)
        else:
            assert self.model.is_partial_fit==True, "it is not partial fit model!"
    
    def set_save_proc(self, save_proc):
        self.save_proc = save_proc
        
    def set_data(self,
                 X_train, 
                 X_test, 
                 y_train, 
                 y_test):
        
        self.X_raw_train = X_train
        self.X_raw_test = X_test
        self.y_raw_train = y_train
        self.y_raw_test = y_test
        
        if self.model.is_use_tensor():
            print("use_tensor")
            self.X_train = data_to_tensor(X_train)
            self.X_test = data_to_tensor(X_test)
            self.y_train = data_to_tensor(y_train)
            self.y_test = data_to_tensor(y_test)

            if self.model.is_use_gpu():
                print("data -> gpu")
                self.X_train = self.X_train.to(self.model.device)
                self.X_test = self.X_test.to(self.model.device)
                self.y_train = self.y_train.to(self.model.device)
                self.y_test = self.y_test.to(self.model.device)
        else:
            print("just set")
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
        
    def close_logger(self):
        if self.logger is not None:
            self.logger.close()
    
    def __del__(self):
        if self.logger is not None:
            self.close_logger()
    
    def shuffle(self, x, y):
        try:
            if self.is_shuffle and model.is_use_gpu() == False:
                x, y = shuffle(x, y)
                print("shuffled!!")
            else:
                x, y = x, y

            return x, y
        except:
            return x, y
    
    def fit(self, ep_log_cycle = 10, is_logging=True):
        print("fit start!")
        self.start_train_time = time.time()
        
        if self.model.is_partial_fit():
            for ep in range(1, self.epoch + 1): # epoch
                X_train, y_train = shuffle(self.X_train, self.y_train)
                
                for i in range(0, len(X_train), self.batch_size): # Mini batch
                    batch_X = X_train[i : i+self.batch_size]
                    batch_Y = y_train[i : i+self.batch_size]
                    self.model.partial_fit(batch_X, batch_Y)
                    """
                    if is_logging:
                        self.logger.add_scalars(str_join([self.log_group, str_join(["batch", "eval"])], "/"), 
                                               {
                                                   "train" : self.train_eval(),
                                                   "test" : self.test_eval(),
                                               },
                                               ep * self.batch_size + i)
                        self.train_evals["batch_train_eval"] = self.train_evals["batch_train_eval"] + [self.train_eval()]
                        self.test_evals["batch_test_eval"] = self.test_evals["batch_test_eval"] + [self.test_eval()]
                        
                        # print("batch acc: ", sum(self.model.predict(self.X_train) == self.y_train) / len(self.y_train))
                    """
                    
                # https://tensorboard-pytorch.readthedocs.io/en/latest/tensorboard.html#tensorboardX.SummaryWriter.add_scalars
                if ep != 1:
                    if is_logging:
                        self.logger.add_scalars(str_join([self.log_group, str_join(["epoch", "eval"])], "/"), 
                                               {
                                                   "train" : self.train_eval(),
                                                   "test" : self.test_eval(),
                                               },
                                               ep)
                        self.train_evals["ep_train_eval"] = self.train_evals["ep_train_eval"] + [self.train_eval()]
                        self.test_evals["ep_test_eval"] = self.test_evals["ep_test_eval"] + [self.test_eval()]
                        
                        if len(self.auxiliary_criterions) > 0:
                            for ax_criterion_name, ax_criterion_func in self.auxiliary_criterions:
                                ax_train_eval = ax_criterion_func(self.predict(self.X_train), self.y_train)
                                self.train_evals[str_join(["ep", "train", ax_criterion_name])] = self.train_evals[str_join(["ep", "train", ax_criterion_name])] + [ax_train_eval]
                                
                                ax_test_eval = ax_criterion_func(self.predict(self.X_test), self.y_test)
                                self.test_evals[str_join(["ep", "test", ax_criterion_name])] = self.test_evals[str_join(["ep", "test", ax_criterion_name])] + [ax_test_eval]
                                
                                self.logger.add_scalars(str_join([self.log_group, str_join([ax_criterion_name, "epoch", "eval"])], "/"), 
                                               {
                                                   "train" : ax_train_eval,
                                                   "test" : ax_test_eval,
                                               },
                                               ep)
                        
                    # Printing
                    if ep % ep_log_cycle == 0:
                        print("ep: ", ep, self.train_eval())
                    
                    if self.save_proc != None:
                        self.save_proc(ep)
        else:
            X_train, y_train = shuffle(self.X_train, self.y_train)
            
            self.model.fit(X_train, y_train)
            
            if is_logging:
                self.logger.add_scalar(str_join([self.log_group, str_join(["train", "eval"])], "/"),
                                       self.train_eval(),
                                       1)

                self.logger.add_scalar(str_join([self.log_group, str_join(["test", "eval"])], "/"),
                                       self.test_eval(),
                                       1)
                
        self.end_train_time = time.time()

        self.train_time = self.end_train_time - self.start_train_time
    
    def predict(self, X):
        return self.model.predict(X)
    
    def eval(self):
        return {
            "train" : self.train_eval(),
            "test" : self.test_eval()
        }
        
    def train_eval(self):
        return self.model.criterion(self.predict(self.X_train), self.y_train).item()
    
    def test_eval(self):
        return self.model.criterion(self.predict(self.X_test), self.y_test).item()
    
    def save(self, save_dir_path, params = {}, post_fix = ""):
        if self.model.is_partial_fit:
            params["epoch"] = self.epoch
            params["batch_size"] = self.batch_size
            
        params["log group"] = self.log_group
                   
        save_m = SaveModel(model=self.model, name=self.name, params=params)
        sj_file_system.save(save_m, os.path.join(save_dir_path, str_join(self.save_file_name, post_fix)))
    
    def evaluate_mode(self):
        try:
            self.model.eval()
        except:
            print("no eval")
   
    def save_total(self, save_dir_path):
        self.close_logger()

        self.X_train = self.X_train.to("cpu")
        self.X_test = self.X_test.to("cpu")
        self.model = self.model.to("cpu")
        
        torch.save(self, os.path.join(save_dir_path, str_join([self.save_file_name])))
            
def df_to_tensor(df):
    return torch.tensor(df.values).float()

def data_to_tensor(data):
    if type(data) == list or type(data) == np.array or type(data) == np.ndarray:
        return torch.tensor(data)
    elif type(data) == pd.DataFrame:
        return df_to_tensor(data)
    elif type(data) == torch.Tensor:
        return data
    
if __name__ == "__main__":
    numerical_gradient(lambda x, a, b: 4 * x ** 2 + a ** 3 + b, [1, 2, 3])

    gradient_descent(lambda a, b: a ** 2 + b ** 2 + b, [3, 3], 0.01, step_num=300)
    
    model = ExecModel(SVC(kernel="linear", C=1.0), 
                      name="SVC_" + kernel + "_C_" + str(C),
                      X_train=x_train_data,
                      X_test=x_test_data,
                      y_train=y_train_data,
                      y_test=y_test_data)