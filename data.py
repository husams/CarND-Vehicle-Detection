import numpy as np
import time
import glob
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import *


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

    cache = glob.glob('cache.p')

    t=time.time()
    if len(cache) == 0:
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
    else:
        cache           = pickle.load(open("cache.p", "rb"))
        car_features    = cache['car_features']
        notcar_features = cache['notcar_features']
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

    return X_train, X_test, y_train, y_test, X_scaler