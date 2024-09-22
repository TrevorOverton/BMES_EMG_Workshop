#train on all and test on all locations, only EMG data 
# imports
import argparse
import itertools
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from scipy.io import loadmat

from TimeDomainFilter import TimeDomainFilter
from LinearDiscriminantAnalysis import LinearDiscriminantAnalysis
import sys

import seaborn as sns
import pandas as pd
import time

from colorama import just_fix_windows_console
from termcolor import colored

from sklearn.metrics import accuracy_score
# parameters
parser = argparse.ArgumentParser()

parser.add_argument( '--emg_window_size', type = int, nargs = '+', action = 'store', dest = 'emg_window_size', default = 50 )
parser.add_argument( '--emg_window_step', type = int, nargs = '+', action = 'store', dest = 'emg_window_step', default = 50 )
args = parser.parse_args()

# command-line parameters
emg_window_size = args.emg_window_size[0] if type( args.emg_window_size ) is list else args.emg_window_size
emg_window_step = args.emg_window_step[0] if type( args.emg_window_step ) is list else args.emg_window_step

CLASSES = [ 'open', 'close', 'flexsion', 'extension', 'rest' ]

# function definitions

def confusion_matrix( ytest, yhat, labels = [], cmap = 'viridis', ax = None, show = True ):
    """
    Computes (and displays) a confusion matrix given true and predicted classification labels

    Parameters
    ----------
    ytest : numpy.ndarray (n_samples,)
        The true labels
    yhat : numpy.ndarray (n_samples,)
        The predicted label
    labels : iterable
        The class labels
    cmap : str
        The colormap for the confusion matrix
    ax : axis or None
        A pre-instantiated axis to plot the confusion matrix on
    show : bool
        A flag determining whether we should plot the confusion matrix (True) or not (False)

    Returns
    -------
    numpy.ndarray
        The confusion matrix numerical values [n_classes x n_classes]
    axis
        The graphic axis that the confusion matrix is plotted on or None

    """
    def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
        """Add a vertical color bar to an image plot."""
        divider = axes_grid1.make_axes_locatable(im.axes)
        width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
        pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
        current_ax = plt.gca()
        cax = divider.append_axes("right", size=width, pad=pad)
        plt.sca(current_ax)
        return im.axes.figure.colorbar(im, cax=cax, **kwargs)

    cm = sk_confusion_matrix( ytest, yhat )
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if ax is None:    
        fig = plt.figure()
        ax = fig.add_subplot( 111 )

    try:
        plt.set_cmap( cmap )
    except ValueError: cmap = 'viridis'

    im = ax.imshow( cm, interpolation = 'nearest', vmin = 0.0, vmax = 1.0, cmap = cmap )
    add_colorbar( im )

    if len( labels ):
        tick_marks = np.arange( len( labels ) )
        plt.xticks( tick_marks, labels, rotation=45 )
        plt.yticks( tick_marks, labels )

    thresh = 0.5 # cm.max() / 2.
    colors = mpl.colormaps[cmap]
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        r,g,b,_ = colors(cm[i,j])
        br = np.sqrt( r*r*0.241 + g*g*0.691 + b*b*0.068 )
        plt.text(j, i, format(cm[i, j], '.2f'),
                    horizontalalignment = "center",
                    verticalalignment = 'center',
                    color = "black" if br > thresh else "white")

    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    ax.set_ylim( cm.shape[0] - 0.5, -0.5 )
    plt.tight_layout()
    if show: plt.show( block = True )
    
    return cm, ax


colors = ['r', 'g', 'b', 'y', 'm']
classes = {'open':0, 'close':1, 'flexion':2, 'extension':3, 'rest':4, '0': 0, '1': 1, '2':2, '3':3, '4': 4, 0:0, 1:1, 2:2, 3:3, 4:4}

acc = []
subplusses = []

session_list = [[[1, 'static'], [2, 'static'], [3, 'dynamic'], [4, 'dynamic'], [5, 'object']],
                [[1, 'static'], [2, 'static'], [3, 'object'], [4, 'dynamic'], [5, 'dynamic']],
                [[1, 'dynamic'], [2, 'dynamic'], [3, 'static'], [4, 'static'], [5, 'object']],
                [[1, 'dynamic'], [2, 'dynamic'], [3, 'object'], [4, 'static'], [5, 'static']],
                [[1, 'object'], [2, 'static'], [3, 'static'], [4, 'dynamic'], [5, 'dynamic']],
                [[1, 'object'], [2, 'dynamic'], [3, 'dynamic'], [4, 'static'], [5, 'static']],
                [[1, 'static'], [2, 'static'], [3, 'dynamic'], [4, 'dynamic'], [5, 'object']],
                [[1, 'static'], [2, 'static'], [3, 'object'], [4, 'dynamic'], [5, 'dynamic']],
                [[1, 'dynamic'], [2, 'dynamic'], [3, 'static'], [4, 'static'], [5, 'object']],
                [[1, 'dynamic'], [2, 'dynamic'], [3, 'object'], [4, 'static'], [5, 'static']],
                [[1, 'object'], [2, 'static'], [3, 'static'], [4, 'dynamic'], [5, 'dynamic']],
                [[1, 'object'], [2, 'dynamic'], [3, 'dynamic'], [4, 'static'], [5, 'static']],
                [[1, 'static'], [2, 'static'], [3, 'dynamic'], [4, 'dynamic'], [5, 'object']],
                [[1, 'static'], [2, 'static'], [3, 'object'], [4, 'dynamic'], [5, 'dynamic']],
                [[1, 'dynamic'], [2, 'dynamic'], [3, 'static'], [4, 'static'], [5, 'object']],
                [[1, 'dynamic'], [2, 'dynamic'], [3, 'object'], [4, 'static'], [5, 'static']],
                [[1, 'object'], [2, 'static'], [3, 'static'], [4, 'dynamic'], [5, 'dynamic']],
                [[1, 'object'], [2, 'dynamic'], [3, 'dynamic'], [4, 'static'], [5, 'static']]]

data = []
reversed_classes = {value: key for key, value in classes.items()}

stype = "static"

for subject_num in range(1, 19):
    for ses_num, ses_type in session_list[subject_num - 1]:
        if ses_type == stype:
            try:
                dataset = np.load(f'data_new/EMGsub{subject_num}_{ses_num}.npy', allow_pickle=True) #returns dict of np.arrays
            except FileNotFoundError:
                continue
            data.append(dataset)
            subplusses.append([subject_num, ses_num])


d = 0
for dataset in data:
    d = d+1
    emg = dataset.item()['emg']
    try:
        labels =np.char.strip(dataset.item()['labels'], ' "\'')
    except TypeError:
        labels = dataset.item()['labels']

    labels = np.array([classes[label] for label in labels])
    locations =np.char.strip(dataset.item()['locations'], ' "\'')
    emg = np.array(emg)
    print(f'the emg data: {emg.shape} , labels: {labels.shape}, locations: {locations.shape}')
    print( 'Done loading the data!' )
    print(f'the data shape is {emg.shape} and labels shape is {labels.shape} and locations shape is {locations.shape}')


    # download training data
    print( 'Importing training data...', end = '', flush = True)
    training_data = emg
    print( 'Done!' )

    # create feature extracting filters
    print( 'Creating EMG feature filters...', end = '', flush = True )
    td5 = TimeDomainFilter()
    print( 'Done!' )

    # compute training features and labels
    print( 'Computing features...', end = '', flush = True)
    num_trials = training_data.shape[0]
    X = []
    y = []
    loc = []
    for j in range( num_trials ):
        trial_data = []
        raw_data = training_data[j]
        num_samples = raw_data.shape[0] 
        label = int(labels[j])
        if stype == "static":
            location = locations[j]
        else:
            location = locations.T[j]
        idx = 0
        while idx + emg_window_size < num_samples:
            window = raw_data[idx:(idx+emg_window_size),:]
            time_domain = td5.filter( window ).flatten()
            trial_data.append( np.hstack( time_domain ) )
            idx += emg_window_step 
        X.append( np.vstack( trial_data ) )
        y.append( label * np.ones( ( X[-1].shape[0], ) ) )
        try:
            loc.append(int(location) * np.ones( ( X[-1].shape[0], ) ) )
        except TypeError:
            loc.append([int(location[0].strip("[]")) * np.ones( ( X[-1].shape[0], ) ), int(location[1].strip("[]")) * np.ones( ( X[-1].shape[0], ) )])
    X = np.vstack( X )
    y = np.hstack( y )
    loc = np.hstack (loc)
    if len(loc) == 2:
        loc = loc.T
    print(f'this is the shape of X {X.shape} and y {y.shape} and location {loc.shape}')
    print( 'Done!' )

    labels_stack = y
    #labels_stack = np.random.shuffle(y) #testing random labels.............................................

    # split data
    print( 'Computing train/test split...', end = '', flush = True )
    Xtrain, Xtest, ytrain, ytest = train_test_split( X, y, test_size = 0.33, random_state = 42, stratify= loc )
    print( 'Done!' )

    # train classifier
    print( 'Training classifier...', end = '', flush = True )
    mdl = LinearDiscriminantAnalysis( Xtrain, ytrain )
    print( 'Done!' )

    # test classifier
    print( 'Testing classifier...', end = '', flush = True )
    yhat = mdl.predict( Xtest )
    print( 'Done!' )

    acc.append(accuracy_score(ytest, yhat, normalize=True))

    #plt.show()
    """
    if d == 1:
        title = 'Sub2_Ses4_all'
    else:
        title = 'Sub2_Ses5_all'
    #plot tsne for the data 
    from sklearn.manifold import TSNE
    X_embedded = TSNE(n_components=2, verbose=True).fit_transform(Xtrans)
    #plt.figure()
    # for i in range(5):
    #     idx = (y == i)
    #     plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], c = colors[i], label = reversed_classes[i])
    #     plt.xlabel('Component 1')
    #     plt.ylabel('Component 2')
    # plt.legend()
    # plt.title(title)
    # plt.show()
    reversed_classes = {0: 'open', 1: 'close', 2: 'flexion', 3: 'extension', 4: 'rest'}

    X_data = pd.DataFrame(X_embedded, columns=['Dim_1', 'Dim_2'])
    time_start = time.time()
    mapped_labels = [reversed_classes[label] for label in labels_stack]
    color_mapping = {
    "open": "purple",
    "close": "orange",
    "flexion": "blue",
    "extension": "red",
    "rest": "black"
    }  
    fig = plt.figure(figsize=(16,10))
    sns.scatterplot(
        x='Dim_1', y='Dim_2',  
        hue=mapped_labels,
        palette=color_mapping,
        data=X_data,
        legend="full",
        alpha=0.9
    )
    plt.legend( prop={'size': 22}, title_fontsize='40', loc = 'lower right')

    plt.title(title)
    plt.tight_layout()
    filename = f'{title}_tsne.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')  
    #plt.show()
    """


for i in range(len(acc)):
    print(colored(f"Subject {subplusses[i][0]}, Session {subplusses[i][1]} accuracy is: {acc[i]}", "green"))

print(colored(f"Mean: {np.mean(acc)}, SD: {np.std(acc)}", "light_red"))

plt.close("all")

plt.boxplot(acc)
plt.violinplot(acc)

plt.show()
