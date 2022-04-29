# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 19:12:48 2019

@author: STU24
"""

# Visualize 관련

# Common Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy
from enum import Enum
from Module import sj_string, sj_file_system

# Custom Libraries

# Sources

class Visualizing(Enum):
    scatter = 1 << 0
    plot = 1 << 1

    sc_pl = scatter | plot

def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys())

def draw_bar_plot(x_list, y_list, title = "Title", xlabel = "xlabel", ylabel = "ylabel"):
    plt.bar(x_list, y_list)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def draw_scatter_plot(x_list, y_list, title = "Title", xlabel = "xlabel", ylabel = "ylabel"):
    plt.scatter(x_list, y_list)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

# 선그래프를 그린다. 이때 주어지는 각 data_sets의 요소로 하나의 선이 구성된다.
# data_sets은 리스트로 주되 Series 형식이어야함 
def draw_line_graph(data_sets, 
                    x_marks = None, 
                    title = None,
                    ylim = None, 
                    xlabel = None, 
                    ylabel = None,
                    legend = None):
    for ds in data_sets:
        ds.index = list(range(0, len(ds)))
        
    # 각 데이터 셋을 하나의 DataFrame으로 만들어서 선그래프를 그리자
    df = pd.concat(data_sets, axis= 1)
    
    fig, ax = plt.subplots(1,1)
    ax.plot(df)
    
    # x축을 조정하고
    if x_marks is not None:
        plt.xticks(df.index, x_marks)
    
    # y축을 조정한다
    if ylim is not None:
        ax.set_ylim(ylim)
    
    if title is not None:
        plt.title(title)
    
    # 축 이름 설정
    if xlabel is not None:
        plt.xlabel(xlabel)
        
    if ylabel is not None:
        plt.ylabel(ylabel)
    
    if legend is not None:
        plt.legend(legend)
        
    return ax

# stack graph를 그림
# data_sets: data_set을 요소로 갖는 리스트, 
# legends: 범례로 들어갈 리스트
# x_marks: x축 눈금에 들어갈 이름
# x_label: x축 이름
# y_label: y축 이름
def draw_stack_graph(data_sets, 
                     legends = None,
                     title = None,
                     x_marks = None,
                     x_rotation = None,
                     x_label = None,
                     y_label = None):
    longest_set_length = 0
    for data_set in data_sets:
        data_length = len(data_set)
        if longest_set_length < data_length:
            longest_set_length = data_length
    
    indexes = range(0, longest_set_length)
    
    width = 0.35
    
    import matplotlib.pyplot as plt
    
    plts = []
    for i, data_set in enumerate(data_sets):
        if i == 0:
            c_plt = plt.bar(indexes, data_set, width)
        else:
            c_plt = plt.bar(indexes, data_set, width, bottom = data_sets[i-1])
        plts.append(c_plt)
    
    ps = [plt[0] for plt in plts]
    
    if legends is not None:
        plt.legend(tuple(ps), tuple(legends))
    
    if title is not None:
        plt.title(title)
    
    if x_rotation is not None and x_marks is not None:
        plt.xticks(indexes, x_marks, rotation = x_rotation)
    else:
        if x_marks is not None:
            plt.xticks(indexes, x_marks)
    
    if x_label is not None:
        plt.xlabel(x_label)
        
    if y_label is not None:
        plt.ylabel(y_label)
    
    return plt

def draw_function(x, function):
    """
    Drawing graph of function

    :param x: numpy array ex) np.linespace(-100, 100, 1000)
    :param function: function ex) lambda x: x+1
    """
    plt.plot(x, list(map(lambda element: function(element), x)))

def f_importances(coef, names):
    imp = coef
    imp, names = zip(*sorted(zip(imp, names)))

    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)


def make_meshgrid(x1, x2, h=.02):
    """
    make meshgrid

    :param x1: data(np.array)
    :param x2: data(np.array)
    :param h: distance

    return X(grid data), Y(grid data)
    """
    x1_min, x1_max = x1.min() - 1, x1.max() + 1
    x2_min, x2_max = x2.min() - 1, x2.max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
    return xx1, xx2


def plot_contours(ax, clf, xx1, xx2, **params):
    """
    plot contour

    :param ax: AxesSubplot
    :param clf: classifier
    :param xx: grid data
    :param yy: grid data
    """
    Z = clf.predict(np.c_[xx1.ravel(), xx2.ravel()])
    Z = Z.reshape(xx1.shape)

    out = ax.contourf(xx1,
                      xx2,
                      Z,  # predict value
                      **params)

    return out

def draw_subplots(figsize, n_cols, draw_functions):
    """
    draw multi subplot
    
    :param figsize: tuple ex) (15,2)
    :param n_cols: set fixed column
    :param draw_functions: drawing function list ex) [ lambda: plt.bar([1],[1]) ]
    """
    plt.figure(figsize=figsize)
    
    while(True):
        draw_count = len(draw_functions)
        if draw_count % n_cols == 0:
            break
        else:
            draw_functions += [lambda: plt.cla()]
            
    
    count = 1
    for draw_function in draw_functions:
        plt.subplot(int(draw_count / n_cols), n_cols, count)
        draw_function()
        count += 1
    plt.tight_layout()

def one_to_many_scatters(data, data_colors, fix_col_name, subplot_size = 4, figsize = (25,25)):
    """
    Drawing many scatter plot representing fix_col to other columns
    
    :param data: (DataFrame)
    :param data_colors: color for visualizing data point ex) ["red", "blue", "red"...]
    :param fix_col_name: x-axis column_name
    :param subplot_size: #suplot_size ex) 4
    :param figsize: figure size
    """
    fig = plt.figure(figsize = figsize)

    for i, colname in enumerate(data.columns):
        if colname == fix_col_name:
            continue
        row = int(i / subplot_size)
        col = i % subplot_size

        ax = fig.add_subplot(subplot_size, subplot_size, row*subplot_size + col + 1)
        ax.scatter(x = data.loc[:,fix_col_name], y = data.loc[:,colname], c=data_labels);
        ax.set(xlabel=fix_col_name, ylabel= colname)
        
def plot_timeseries(data, labels=None, linewidth=3):
    '''Plot a timeseries
    
    Args:
        data: (np.ndarray) signal varying over time, where each column is a different signal.
        labels: (list) labels which need to correspond to the number of columns.
        linewidth: (int) thickness of line
    '''
    plt.figure(figsize=(20,5))
    plt.plot(data, linewidth=linewidth)
    plt.ylabel('Intensity', fontsize=18)
    plt.xlabel('Time', fontsize=18)
    plt.tight_layout()
    if labels is not None:
        if len(labels) != data.shape[1]:
            raise ValueError('Need to have the same number of labels as columns in data.')
        plt.legend(labels, fontsize=18)

def plot_design_matrix(design_mat_array, regressor_indexes, figsize = (6,10), regressor_labels=None):
    """
    Plotting Design matrix
    
    :param design_mat_array: design matrix(numpy array). columns: regressor, rows: data
    :param regressor_indexes: regressor index for plotting(list) ex) [1,2,3]
    :param figsize: figure size(tuple)
    :param regressor_labels: regressor labels for showing to title
    """
    X = design_mat_array
    
    data_length = X.shape[0]
    
    f, a = plt.subplots(ncols = len(regressor_indexes), figsize=figsize, sharey=True)
    plt.gca().invert_yaxis()
    
    for axis_i in range(len(regressor_indexes)):
        regressor_index = regressor_indexes[axis_i]
        a[axis_i].plot(X[:,regressor_index], range(data_length))
        
        if regressor_labels != None:
            regressor_label = regressor_labels[axis_i]
            a[axis_i].set_title(regressor_label, rotation = 45, fontsize=20)
    plt.tight_layout()

def draw_regPlot(ax, xs, ys, title = None, xlabel = None, ylabel = None):
    """
    Draw regression plot
    
    :param ax: axis
    :param xs: data(list)
    :param ys: data(list)
    :param title: title(string)
    :param xlabel: xlabel(string)
    :param ylabel: ylabel(string)
    """
    sns.regplot(x = xs, 
                y = ys,
                ax = ax)

    print("corr with p: ", scipy.stats.pearsonr(xs, ys))
    
    if xlabel != None:
        ax.set_xlabel(xlabel)
    if ylabel != None:
        ax.set_ylabel(ylabel)
    if title != None:
        ax.set_title(title)

def plot_accuracies(axis,
                    names, 
                    accuracies,
                    threshold = None, 
                    search_names = [], 
                    rank_ranges = None, 
                    exclude_names = [],
                    ylim = None,
                    y_interval = None,
                    xlabel = "design_property",
                    ylabel = "accuracies"):
    """
    Plot decoding results
    
    :param names: names(list)
    :param accuracies: accuracies(2d - list)
    :param threshold: threshold
    :param search_names: search key for searching keyword in roi_decoding_results
    :param ylim: ylim(tuple - y_min, y_max)
    :param y_interval: plotting interval(float)
    
    """
    name_index = 0
    accuracy_index = 1
    
    results = list(zip(names, accuracies))
    
    filtered_results = list(filter(lambda result: sj_string.search_string(target = result[name_index], 
                                                            search_keys = search_names, 
                                                            search_type="all",
                                                                         exclude_keys = exclude_names), 
                                   results))
    
    
    # Sort by accuracy
    filtered_results.sort(key=lambda x: np.mean(x[1]), reverse=True)
    
    # ranging rank
    if rank_ranges != None:
        filtered_results = filtered_results[rank_ranges[0]: rank_ranges[1]]

    # bat plot
    xmin = 0
    xmax = len(filtered_results)
    
    x_data = np.arange(xmin, xmax)
    y_data = [accs for _, accs in filtered_results]

    sns.barplot(data = y_data)
    
    # x-axis
    axis.set_xticks(x_data, [name for name, _ in filtered_results], rotation=90)
    
    # threshold
    bar_width = 0.5

    x_lowerlim = xmin - bar_width
    x_upperlim = xmax - bar_width*(3/2)
    
    if threshold != None:
        axis.plot([x_lowerlim, x_upperlim], [threshold, threshold], "k--")
    
    # y-axis
    if ylim != None:
        axis.set_ylim(ylim)
    
        if y_interval != None:
            axis.set_yticks(np.arange(ylim[0], ylim[1], y_interval))

    # label
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def plot_stats(axis,
               names, 
               stats,
               threshold = None, 
               search_names = [], 
               rank_ranges = None, 
               exclude_names = [],
               ylim = None,
               y_interval = None):
    """
    Plot stats
    
    :param names: names(list)
    :param stats: stats(1d - list)
    :param threshold: threshold
    :param search_names: search key for searching keyword in roi_decoding_results
    :param ylim: ylim(tuple - y_min, y_max)
    :param y_interval: plotting interval(float)
    
    """
    name_index = 0
    accuracy_index = 1
    
    results = list(zip(names, stats))
    
    filtered_results = list(filter(lambda result: sj_string.search_string(target = result[name_index], 
                                                            search_keys = search_names, 
                                                            search_type="all",
                                                                         exclude_keys = exclude_names), 
                                   results))
    
    
    # Sort by accuracy
    filtered_results.sort(key=lambda x: x[1], reverse=True)
    
    # ranging rank
    if rank_ranges != None:
        filtered_results = filtered_results[rank_ranges[0]: rank_ranges[1]]

    # bat plot
    xmin = 0
    xmax = len(filtered_results)
    
    x_data = np.arange(xmin, xmax)
    y_data = [stat for _, stat in filtered_results]

    sns.barplot(x = x_data, y = y_data)
    
    # x-axis
    axis.set_xticks(x_data, [name for name, _ in filtered_results], rotation=90)
    
    # y-axis
    
    if ylim != None:
        axis.set_ylim(ylim)
    
        if y_interval != None:
            axis.set_yticks(np.arange(ylim[0], ylim[1], y_interval))
    
    # threshold
    bar_width = 0.5

    x_lowerlim = xmin - bar_width
    x_upperlim = xmax - bar_width*(3/2)
    
    if threshold != None:
        axis.plot([x_lowerlim, x_upperlim], [threshold, threshold], "k--")
    
    plt.xlabel("design property")
    plt.ylabel("stats")

def draw_plot_box(axis, names, accuracies, search_names = [], exclude_names = [], rank_ranges = (0, 10)):
    """
    Draw box pot
    
    :param names: name of accuracy(list - string) ex) ["hippocampus", "gyrus", "cerebellum"]
    :param accuracies: accuracy(list) ex) [0.1, 0.1, 0.1]
    :param search_names: search name(list - string) ex) ["hippocampus"]
    :param exclude_names: exclude name(list - string) ex) ["hippocampus"]
    :param rank_ranges: range(tuple) ex) (0, 10)
    """
    results = list(zip(names, accuracies))
    rank_ranges = (0, 10)
    
    name_index = 0
    filtered_results = list(filter(lambda result: sj_string.search_string(target = result[name_index], 
                                                            search_keys = search_names, 
                                                            search_type="all",
                                                                         exclude_keys = exclude_names), 
                                   results))


    # Sort by accuracy
    filtered_results.sort(key=lambda x: np.mean(x[1]), reverse=True)

    # ranging rank
    if rank_ranges != None:
        filtered_results = filtered_results[rank_ranges[0]: rank_ranges[1]]

    
    
    # data
    target_names = list(map(lambda result: result[0], filtered_results))
    y_data = [accs for _, accs in filtered_results]
    
    # plot
    axis.boxplot(y_data)
    
    axis.set_xticks(np.arange(1, len(target_names) + 1), target_names, rotation=90)
    
    # label
    axis.set_xlabel("design property")
    axis.set_ylabel("accuracy")
    
    # Title
    if len(search_names) != 0:
        search_title = "search: " + sj_file_system.str_join(search_names)
    else:
        search_title = "search: All"
    
    if len(exclude_names) != 0:
        exclude_title = "exclude: " + sj_file_system.str_join(exclude_names)
    else:
        exclude_title = ""
    range_title = "range: " + str(rank_ranges[0]) + " ~ " + str(rank_ranges[1])

    title = sj_file_system.str_join([search_title, exclude_title, range_title], ", ")
    axis.set_title(title)
    
    
if __name__=="__main__":
    import F_Visualize
    test = pd.DataFrame([
        [100, 200, 150],
        [123, 180, 159],
        [130, 190, 182],
        [134, 210, 167],
        [159, 230, 171],
        [160, 235, 180],
        [169, 237, 188]
                ])

    a = F_Visualize.draw_line_graph([test[0], test[1], test[2]], x_marks = ['아', '야', '어', '여', '오', '요'],
                                    ylim = [0, 300],
                                    xlabel = 'x축',
                                    ylabel = 'y축',
                                    title = 'abc')
    
    F_Visualize.draw_stack_graph([[1, 2, 3, 4], [5, 6, 7, 8]],
                                 legends = ['1234','456'],
                                 title = '1234',
                                 x_marks = ['가','나','다','라'],
                                 x_label = 'Y축!~',
                                 y_label = 'X축!~')

    F_Visualize.draw_function(np.linspace(0, 100, 100), lambda x: x + 1)

    data = {
        "a": [1, 2, 3, 2, 1],
        "b": [2, 3, 4, 3, 1],
        "c": [3, 2, 1, 4, 2],
        "d": [5, 9, 2, 1, 8],
        "e": [1, 3, 2, 2, 3],
        "f": [4, 3, 1, 1, 4],
    }

    fig, ax = plt.subplots()
    bar_plot(ax, data, total_width=.8, single_width=.9)
    plt.show()

    from sklearn import svm
    fig, ax = plt.subplots()
    xx1, xx2 = make_meshgrid(np.array([0, 1]), np.array([1, 2]), h=0.5)
    clf = svm.SVC(kernel='linear')
    plot_contours(ax, clf, xx1, xx2, cmap=plt.cm.coolwarm, alpha=0.8)

    draw_subplots((15,2), 3, [lambda: plt.bar([1],[1]), lambda: plt.bar([1],[1]), lambda: plt.bar([1],[1]), lambda: plt.bar([1],[1])])
    
    one_to_many_scatters(data = pd.DataFrame({
        "A" : [1,2,3],
        "B" : [4,5,6],
        "C" : [1,1,1]
    }),
                         data_labels = ["red", "blue", "red"],
                         fix_col_name = "A")

    plot_timeseries(np.c_[
        [1,2,3],
        [4,5,6]
    ])
    
    fig, axis = plt.subplots(1,1)
    draw_plot_box(axis, ["a", "b"], [[1,2,3], [4,5,6]])
    
    fig, axis = plt.subplots(1,1)
    plot_accuracies(axis, ["a", "b"], [[1,2,3], [4,5,6]])
    
    fig, axis = plt.subplots(1,1)
    plot_stats(axis, ["a", "b"], [1,2])
    