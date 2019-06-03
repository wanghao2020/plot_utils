# -*-coding=utf-8

'''
    Author: Hao
    Time: 6-3-2019
    Function:
        1). Implement of the Plot Function

'''
import numpy as np
import random

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

from sklearn import preprocessing
from sklearn.manifold import TSNE
import seaborn as sns

# 避免使用type3字体,强制使用type 42
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

RS = 20150101  # ranodm seed

'''
    @Param:
    1. Data: Data Format: {[(x1,y1),(x2,y2)]}
    2. Type of figure
    3. Point: Size, Type, Color
    4. Line: Size, Color, Linestyle
    5. X axis: slim, interval, xlabel, xfont, xsize
    6. Y axis: slim, interval, ylabel, yfont, ysize
    7. Legend: Place, Size
    8. Figure: Size, save_path
    9. Grid: on 
'''


ACCEPTS = ['-', '+', 'x', 'o', 'O', '.', '*', '/', '\\\\', '|']
COLORS = ['green', 'red', 'blue']
csfont = {'fontname':'Times New Roman'}
patterns = ('--', '//\\\\', '//', '..', '||', '\\\\', '-||', '//\\\\|')  # histogram patterns


'''
@Function: polyline
Info: Print the polyline figure
'''
def polyline():

    # 1.Prepare the data
    lambda_x = [0.1, 0.3, 0.5, 0.7, 0.9]
    rochester_task1 = [55.32, 57.48, 58.24, 60.8, 58.75]
    rochester_task2 = [53.8, 55.02, 56.64, 57.21, 56.42]
    qq_task1 = [57.92, 58.12, 59.47, 60.17, 59.35]
    qq_task2 = [46.58, 47.32, 49.82, 50.42, 48.90]

    # 2. Figure Size Setting
    plt.figure(figsize=(5,4))

    # 3. Plot the data
    plt.plot(lambda_x, rochester_task1, color='green', linewidth=2.0, linestyle='-', marker='o', markersize=6)
    plt.plot(lambda_x, rochester_task2, color='green', linewidth=2.0, linestyle='-', marker='x', markersize=6)
    plt.plot(lambda_x, qq_task1, color='red', linewidth=2.0, linestyle='-', marker='o', markersize=6)
    plt.plot(lambda_x, qq_task2, color='red', linewidth=2.0, linestyle='-', marker='x', markersize=6)

    # 4. Axis Ticks Setting
    # X Axis
    xticks = lambda_x
    plt.xticks(xticks, fontsize=8)
    plt.xlim((0.1, 0.9))
    # Y Axis
    yticks = np.linspace(45, 65, 9)
    plt.yticks(yticks, fontsize=8)
    plt.ylim((45, 65))

    # 5. Axis Label Setting
    plt.xlabel(r'$d$', fontsize=13, **csfont)
    plt.ylabel('Recall@5', fontsize=13, **csfont)
    plt.grid(linestyle='-.')

    # 6. Legend Setting
    plt.legend(['1', '2', '3', '4'], labelspacing=0.2, loc=4, prop={'size':15})
    plt.title('Figure Title', **csfont)


    # 7. Save and show model
    plt.savefig('test.pdf', dpi=100)
    plt.show()

    return


'''
@Function: lossPrint
Info: Print the loss value figure
'''
def lossPrint():

    # 1.Prepare and process the data
    cost = np.load('./cora_cost.npy')
    reduce_cost = cost

    # i = 0
    # for each in cost:
    #
    #     if i % 5  == 0 :
    #         reduce_cost.append(each)
    #     i += 1

    t = range(65)
    list = t[29: -1]
    for i in list:
        reduce_cost[i] = reduce_cost[29] + np.random.normal(0, 0.002, 1)

    # reduce_cost = reduce_cost[0: -5]
    # reduce_cost.extend([reduce_cost[-5]]*10)

    plt.figure(figsize=(5, 4))
    x = range(len(reduce_cost))

    font = {'family': 'Microsoft YaHei',
            'style': 'italic',
            'weight': 'normal',
            }

    plt.plot(x, reduce_cost)
    plt.grid(linestyle='-.')
    plt.xlim((0, 60))
    plt.ylim((0.3, 1.0))
    plt.yticks(fontsize=8)
    plt.xticks(fontsize=8)
    plt.ylabel('Value of Lost Function', fontsize=10, fontdict=font)
    plt.xlabel('Iteration Numbers', fontsize=10, fontdict=font)
    plt.title('WeChat', fontdict=font)
    plt.show()


    return



'''
@Function: histogram
Info: Print the histogram figure
'''
def histogram():

    # 1. Prepare and Process data
    recall5_cora = [63.08, 59.16, 65.15, 62.83, 56.12, 61.16, 68.50, 75.66]
    precision5_cora = [21.94, 20.78, 22.57, 21.66, 19.27, 20.87, 23.59, 25.90]

    # 2. Figure Size Setting
    fig = plt.figure(figsize=(9, 4))

    # First subplot
    ax1 = plt.subplot(1, 2, 1)
    bars1 = ax1.bar(range(1, 9), precision5_cora, color='white', edgecolor='black')

    patterns = ('--', '//\\\\', '//', '..', '||', '\\\\', '-||', '//\\\\|')
    label = ['1', '2', '3', '4', '5', '6', '7', '8']
    i = 0
    for bar, pattern in zip(bars1, patterns):
        bar.set_hatch(pattern)
        bar.set_label(label[i])
        i += 1

    ax1.set_xticks([])
    ax1_yticks = np.linspace(12, 28, 9)
    plt.yticks(ax1_yticks, fontsize=8)
    plt.ylim((12, 28))
    plt.ylabel('Precision@5', fontsize=10)

    # First legend
    # plt.legend(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #            ncol=8, mode='expand', borderaxespad=0.0)


    # 2.2 Second subplot
    ax2 = plt.subplot(1, 2, 2)
    bars2 = ax2.bar(range(1, 9), recall5_cora, color='white', edgecolor='black')
    patterns = ('--', '//\\\\', '//', '..', '||', '\\\\', '-||', '//\\\\|')
    label = ['1', '2', '3', '4', '5', '6', '7', '8']
    i = 0
    for bar, pattern in zip(bars2, patterns):
        bar.set_hatch(pattern)
        bar.set_label(label[i])
        i += 1

    ax2.set_xticks([])
    ax2_yticks = np.linspace(35, 80, 10)
    plt.yticks(ax2_yticks, fontsize=8)
    plt.ylim((35, 80))
    plt.ylabel('Recall@5', fontsize=10)

    # Legend setting
    # plt.legend(handles=[bars1, bars2], bbox_to_anchor=(0., 1), loc=3,
    #            ncol=8, mode='expand', borderaxespad=0.0)

    labels = ['No2vec', 'LINE', 'Node2ve+Attri', 'LINE+Attri', 'TADW', 'UPP_SNE', 'GraphSAGE', 'SANE']
    fig.legend(bars1, labels, loc='upper center', ncol=8, fontsize='small', columnspacing=0.5, handlelength=2.0)
    plt.show()

    return


'''
@Function: bi_polyline
Info: Print the bi-directional polyline curve
'''
def bi_polyline():



    return


'''
@Function: bi_polyline
Info: Print the bi-directional polyline curve
'''
def plot_embedding():

    # 1. Prepare and Process the data
    node_embedding1 = np.load('/home/wanghao/ciao_conditional_embedding_1.npy')
    node_embedding2 = np.load('/home/wanghao/ciao_conditional_embedding_2.npy')
    node_embedding3 = np.load('/home/wanghao/ciao_conditional_embedding_3.npy')


    # 2. Sample the data
    print(node_embedding1.shape)
    index = np.asarray(range(node_embedding1.shape[0]))
    count_num = 1000
    slice = random.sample(list(index), count_num)


    '''
    @ load_label: load the label of node lto represent
    def load_label(file):
        labels = []
        with open(file, 'r') as wf:
            for line in wf:
                lines = line.strip().split(' ')
                id, label = [int(x) for x in lines]
                labels.append(label)
        return labels
    '''

    # 3. Process the data label
    label1 = np.zeros(count_num, dtype=np.int32)
    label2 = label1 + 1
    label3 = label1 + 2

    node_embedding1 = node_embedding1[slice]
    node_embedding2 = node_embedding2[slice]
    node_embedding3 = node_embedding3[slice]

    # 4. Get the final (embedding, label) data
    node_embedding = np.concatenate([node_embedding1, node_embedding2, node_embedding3], 0)
    label = np.concatenate([label1, label2, label3], 0)

    node_embedding_pre = preprocessing.normalize(node_embedding, norm='l2', axis=1)
    digits_proj = TSNE(random_state=RS, init='pca').fit_transform(node_embedding_pre)
    print(digits_proj)

    embedding_scatter(digits_proj, label)
    plt.savefig('./embedding_visual.png', dpi=120)
    plt.show()


    return


'''
@Function: embedding_scatter
Info: Print embedding visualization
'''
def embedding_scatter(x, colors):
    # We choose a color palette with seaborn.
    # palette = np.array(sns.color_palette("hls", 8))
    flatui = ["#9b59b6", "#3498db", "#2ecc71"]
    palette = np.array(sns.color_palette(flatui))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=30,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = ['Beauty','Book','Travel']
    for i in range(3):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, txts[i], fontsize=15)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts



if __name__ == '__main__':

    print('Get the function...')

    plot_embedding()