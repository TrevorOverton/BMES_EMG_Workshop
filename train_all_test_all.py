#Classify forearm EMG signals with an LDA model
import numpy as np
from sklearn.model_selection import train_test_split
from TimeDomainFilter import TimeDomainFilter
from LinearDiscriminantAnalysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

#Variables
emg_window_size = 50
emg_window_step = 50
classes = {'open':0, 'close':1, 'flexion':2, 'extension':3, 'rest':4} #dictionary for mapping labels to numbers

dataset = np.load(f'example_data.npy', allow_pickle=True) #opens data file, returns dict of np.arrays

#get only EMG data from dataset
emg = dataset.item()['emg'] 
emg = np.array(emg)

#get label information from dataset
labels =np.char.strip(dataset.item()['labels'], ' "\'')
labels = np.array([classes[label] for label in labels])

#get location information from dataset
locations =np.char.strip(dataset.item()['locations'], ' "\'')

#print shape of data to make sure everything went smoothly
print(f'the emg data: {emg.shape} , labels: {labels.shape}, locations: {locations.shape}')

#Create variables for loop
num_trials = emg.shape[0]
X = [] #emg data
y = [] #label data
loc = [] #location data
td = TimeDomainFilter() #TD filter

#Iterate through trials, divide into windows and send to TD filter
for j in range( num_trials ):
    trial_data = []
    raw_data = emg[j]
    label = int(labels[j])
    location = locations[j]
    idx = 0
    while idx + emg_window_size < 2500:
        window = raw_data[idx:(idx+emg_window_size),:]
        time_domain = td.filter(window).flatten()
        trial_data.append( np.hstack( time_domain ) )
        idx += emg_window_step 
    X.append( np.vstack( trial_data ) )
    y.append( label * np.ones( ( X[-1].shape[0], ) ) )
    loc.append(int(location) * np.ones( ( X[-1].shape[0], ) ) )

#Organize data, print to make sure shape is good
X = np.vstack(X)
y = np.hstack(y)
loc = np.hstack(loc)
print(f'this is the shape of X {X.shape} and y {y.shape} and location {loc.shape}')

#Split data into training and testing datasets
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.33, random_state=42, stratify=loc)

#Train classifier
mdl = LinearDiscriminantAnalysis(Xtrain, ytrain)

#Test classifier
yhat = mdl.predict(Xtest)

print(accuracy_score(ytest, yhat, normalize=True))
