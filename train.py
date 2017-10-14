import numpy as np
import glob
import time
import pickle
import argparse
from sklearn.svm import LinearSVC
from data import preprocess_data


def fit_model(config):
    # Images files
    cars    = glob.glob('./vehicles/*/*.png')
    notcars = glob.glob('./non-vehicles/*/*.png')

    # Load imagea nd extract features
    X_train, X_test, y_train, y_test, X_scaler = preprocess_data(cars, notcars, config)

    print('Using:',config["color_space"], "color space", config["orient"],'orientations',config["pix_per_cell"],
         'pixels per cell and', config["cell_per_block"],'cells per block')
    print('Feature vector length:', len(X_train[0]))

    with open('dataset.p', 'wb') as out:
        pickle.dump({"X_train": X_train,
                     "y_trian": y_train,
                     "X_test":  X_test,
                     "y_test": y_test}, out)
    # Create SVM leab=ner model
    svc = LinearSVC()

    # Train model
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    # Store scale and trained model
    config["svc"]    = svc
    config["scaler"] = X_scaler

    return config

def str2Channel(value):
    """
        Parse HOG channel argument
    input:
        value(str): Argument value, should be All or (0,1,2)
    Returns:
        All or integer value
    """
    try:
        # return ALL or try and convert to integer
        return value if value == "ALL" else int(value)
    except ValueError:
        raise argparse.ArgumentTypeError('All or integer value (0-2) expected.')
    

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train liner SVM model .')
    parser.add_argument('--model', dest="model", type=str, default="svn", help='Model name')
    parser.add_argument('--color_space', dest="color_space", type=str, default="HSV", help='color')
    parser.add_argument('--spatial_size', dest="spatial_size", type=int, default=16, help='spatial size')
    parser.add_argument('--hist_bins', dest="hist_bins", type=int, default=16, help="histogram bins")
    parser.add_argument('--orient', dest="orient", type=int, default=9, help="orientations")
    parser.add_argument('--pix_per_cell', dest="pix_per_cell", type=int, default=8, help="pixel per cell")
    parser.add_argument('--cell_per_block', dest="cell_per_block", type=int, default=2, help="cell per block")
    parser.add_argument('--hog_channel', dest="hog_channel", type=str2Channel, default="ALL", help="Hog channel")
    parser.add_argument('--no-spatial_feat', dest='spatial_feat', default=True, action='store_false')
    parser.add_argument('--no-hog_feat', dest='hog_feat', default=True, action='store_false')
    parser.add_argument('--no-hist_feat', dest='hist_feat', default=True, action='store_false')
    args = parser.parse_args()

    # Build model configurtion
    config = {
        "model":            args.model,
        "color_space":      args.color_space,
        "spatial_size":     (args.spatial_size,args.spatial_size),
        "hist_bins":        args.hist_bins,
        "orient":           args.orient,
        "pix_per_cell":     args.pix_per_cell,
        "cell_per_block":   args.cell_per_block,
        "hog_channel":      args.hog_channel,
        "spatial_feat":     args.spatial_feat,
        "hog_feat":         args.hog_feat,
        "hist_feat":        args.hist_feat
    }

    return config


def main():
    # 1. parse arguments
    config = parse_arguments()

    # 2. train model and save
    model = fit_model(config)

    # 3. save model
    with open("{}.p".format(config["model"]),"wb") as file:
        pickle.dump(config, file)
    

if __name__ == '__main__':
    main()