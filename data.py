import numpy as np
import time
import glob
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import *

def has_config_changed(config):
    try:
        # Open file
        with open("{}.p".format(config['model']),'rb"') as file:
            current = pickle.load(file)

        for key in config.keys():
            # Ignore batch size, learning rate and epochs
            if key == 'learning_rate' or key == 'batch_size' or key == 'epochs':
                continue
            if config[key] != current[key]:
                return True
    except:
        return True
    return False

def preprocess_data(cars,not_cars, config):
    color_space    = config['color_space']
    spatial_size   = config["spatial_size"]
    hist_bins      = config["hist_bins"]
    orient         = config["orient"] 
    pix_per_cell   = config["pix_per_cell"] 
    cell_per_block = config["cell_per_block"]
    hog_channel    = config["hog_channel"] 
    spatial_feat   = config["spatial_feat"]
    hist_feat      = config["hist_feat"] 
    hog_feat       = config["hog_feat"]

    t=time.time()
    if has_config_changed(config):
        car_features = extract_features(cars, color_space, spatial_size,
                                        hist_bins, orient, 
                                        pix_per_cell, cell_per_block, hog_channel,
                                        spatial_feat, hist_feat, hog_feat)

        notcar_features = extract_features(not_cars, color_space, spatial_size,
                                        hist_bins, orient, 
                                        pix_per_cell, cell_per_block, hog_channel,
                                        spatial_feat, hist_feat, hog_feat)
        pickle.dump({'car_features': car_features,
                    'notcar_features': notcar_features}, open("cache.p", "wb"))
    
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to extract features ...')

        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state, stratify=y)

        # Validation set to be used with feed forrward neral networks
        X_test, X_valid, y_test, y_valid = train_test_split(
            X_test, y_test, test_size=0.1, random_state=rand_state, stratify=y_test)
        
        dataset = {'X_train' : X_train,
                   'X_test'  : X_test,
                   'X_valid' : X_valid,
                   'y_train' : y_train,
                   'y_test'  : y_test,
                   'y_valid' : y_valid,
                   'X_scaler': X_scaler}

        # Save dataset
        with open("dataset.p",'wb') as file:
            pickle.dump(dataset, file)
    else:
        #Load existing dataset
        with open('dataset.p', 'rb') as file:
            dataset = pickle.load(file)


    return dataset['X_train'], dataset['X_test'], dataset['X_valid'], dataset['y_train'], dataset['y_test'], dataset['y_valid'], dataset['X_scaler']