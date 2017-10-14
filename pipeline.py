import pickle
import cv2
from scipy.ndimage.measurements import label
from utils import get_color_converter, color_hist, bin_spatial
from keras.models import load_model

class VehicleDetector(object):
    def __init__(self, model, threshold=2):
        # Load model
        with open("{}.p".format(model), "rb") as file:
            config = pickle.load(file)
        
        self.model          = load_model('{}.h5'.format(model)) # use Feed forward neural network (Getting better result)
        self.X_scaler       = config["scaler"]
        self.orient         = config["orient"]
        self.pix_per_cell   = config["pix_per_cell"]
        self.cell_per_block = config["cell_per_block"]
        self.spatial_size   = config["spatial_size"]
        self.hist_bins      = config["hist_bins"]
        self.color_space    = config['color_space']
        self.hog_channel    = config['hog_channel']
        self.spatial_feat   = config['spatial_feat']
        self.hist_feat      = config['hist_feat']
        self.hog_feat       = config['hog_feat']

        
        self.current_frame  = 0
        self.heatmap        = None
        self.threshold      = threshold
        self.convert_color  = get_color_converter(self.color_space)
        self.regions        =  [([400,480],(85, 80), (0.75, 0.75)),([380,530], (130, 130), (0.75, 0.75)),
                                ([410,530],(95, 95), (0.5, 0.5)), ([400,600], (230, 100), (0.1, 0.1))]


    def single_img_features(self, image):    
        #1) Define an empty list to receive features
        img_features = []
        feature_image = self.convert_color(image)
        
        #3) Compute spatial features if flag is set
        if self.spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=self.patial_size)
            #4) Append features to list
            img_features.append(spatial_features)
        #5) Compute histogram features if flag is set
        if self.hist_feat == True:
            hist_features = color_hist(feature_image, nbins=self.hist_bins)
            #6) Append features to list
            img_features.append(hist_features)
        #7) Compute HOG features if flag is set
        if self.hog_feat == True:
            if self.hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.extend(self.get_hog_features(feature_image[:,:,channel]))      
            else:
                hog_features = self.get_hog_features(feature_image[:,:,self.hog_channel])

            #8) Append features to list
            img_features.append(hog_features)

        #9) Return concatenated array of features
        return np.concatenate(img_features)

    def search_windows(self, image, windows):

        #1) Create an empty list to receive positive detection windows
        on_windows = []
        #2) Iterate over all windows in the list
        for window in windows:
            #3) Extract the test window from original image
            test_img = cv2.resize(image[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
            #4) Extract features for that window using single_img_features()
            features = self.single_img_features(test_img)
            #5) Scale extracted features to be fed to classifier
            test_features = self.X_scaler.transform(np.array(features).reshape(1, -1))
            #6) Predict using your classifier
            prediction = self.model.predict(test_features)[0,0]
            #7) If positive (prediction == 1) then save the window
            if prediction >= 0.5:
                on_windows.append(window)
        #8) Return windows for positive detections
        return on_windows


    def apply_heat(self, image, heatmap, boxes):
        for box in boxes:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # return new heatmap
        return heatmap

    def draw_labeled_bboxes(self, image, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(image, bbox[0], bbox[1], (0,0,255), 6)
        # Return the image
        return image
    
    def get_hog_features(self,image):
        features = hog(image, orientations=self.orient, 
                       pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                       cells_per_block=(self.cell_per_block, self.cell_per_block), 
                       transform_sqrt=False, 
                       visualise=False, feature_vector=False)
        return features

    def slide_window(self, image, y_start_stop=[None, None], 
                     xy_window=(64, 64), xy_overlap=(0.5, 0.5)):

    
        x_start_stop = [0, image.shape[1]]
        # If y start/stop positions not defined, set to image size
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = image.shape[0]
        # Compute the span of the region to be searched    
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
        ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
        nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
        ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs*nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys*ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]
                
                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows
        return window_list

    def find_car(self, image, y_start_stop=[None, None], 
                     xy_window=(64, 64), xy_overlap=(0.5, 0.5)):

        windows = self.slide_window(image, 
                                    y_start_stop=y_start_stop, 
                                    xy_window=xy_window, xy_overlap=xy_overlap)
        return self.search_windows(image, windows)
        

    def mark_cars(self, frame):
        self.current_frame += 1

        if (self.heatmap  is None) or (self.current_frame % 10 == 0):
            # Create new heatmap.
            heatmap = np.zeros_like(frame[:,:,0]).astype(np.float)
            for y_start_stop, xy_window, xy_overlap in self.regions:
                # Find cars in  the frame
                boxes = self.find_car(frame, y_start_stop, xy_window, xy_overlap)
                # Apply heat map to remove false positive
                heatmap = self.apply_heat(frame, heatmap, boxes)
            # Zero out pixels below the threshold
            heatmap[heatmap <= self.threshold] = 0
            # store new map
            self.heatmap = heatmap
        # create label from the heatmap
        labels = label(self.heatmap)
        
        cv2.putText(frame, "frame: {}".format(self.current_frame), (20,60),  cv2.FONT_HERSHEY_DUPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
        # Draw new labels
        return self.draw_labeled_bboxes(frame, labels)