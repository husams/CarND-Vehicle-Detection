import pickle
import cv2
from scipy.ndimage.measurements import label
from utils import get_color_converter, color_hist, bin_spatial

class VehicleDetector(object):
    def __init__(self, model, threshold=2):
        # Load model
        with open("{}.p".format(model), "rb") as file:
            model = pickle.load(file)
        
        self.svc            = model["svc"]
        self.X_scaler       = model["scaler"]
        self.orient         = model["orient"]
        self.pix_per_cell   = model["pix_per_cell"]
        self.cell_per_block = model["cell_per_block"]
        self.spatial_size   = model["spatial_size"]
        self.hist_bins      = model["hist_bins"]
        self.color_space    = model['color_space']
        self.hog_channel    = model['hog_channel']

        
        self.current_frame  = 0
        self.heatmap        = None
        self.threshold      = threshold
        self.convert_color  = get_color_converter(self.color_space)
        #self.windows        = [(400,500,1), (400,600,1.5) ,(480,656,2.),(480,656,2.5)]
        #self.windows        = [(380,600,1),(400,656,1.5),(400,656,2.0),(400,656,2.5)]
        self.windows        = [(360, 560, 1.5), (400, 600, 1.8), (440, 700, 2.5)]
    
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

    def find_car(self, image, ystart, ystop, scale):
        rectangles = []
        img        = image.astype(np.float32)/255
        
        img_tosearch = img[ystart:ystop,:,:]
        ctrans_tosearch = self.convert_color(img_tosearch)
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
            
        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // self.pix_per_cell) - self.cell_per_block + 1
        nyblocks = (ch1.shape[0] // self.pix_per_cell) - self.cell_per_block + 1
        nfeat_per_block = self.orient * self.cell_per_block ** 2
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window  = (window // self.pix_per_cell) - self.cell_per_block + 1
        cells_per_step      = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        
        # Compute individual channel HOG features for the entire image
        hog1 = self.get_hog_features(ch1)
        hog2 = self.get_hog_features(ch2)
        hog3 = self.get_hog_features(ch3)
        
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1    = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat2    = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3    = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos*self.pix_per_cell
                ytop = ypos*self.pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window],(64,64))
            
                # Get color features
                spatial_features = bin_spatial(subimg, size=self.spatial_size)
                hist_features    = color_hist(subimg, nbins=self.hist_bins)

                # Scale features and make a prediction
                test_features = self.X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
                test_prediction = self.svc.predict(test_features)
                
                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    rectangles.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart))) 
                    
        return rectangles
        

    def mark_cars(self, frame):
        self.current_frame += 1

        if (self.heatmap  is None) or (self.current_frame % 10 == 0):
            # Create new heatmap.
            heatmap = np.zeros_like(frame[:,:,0]).astype(np.float)
            for ystart, ystop, scale in self.windows:
                # Find cars in  the frame
                boxes = self.find_car(frame,ystart,ystop, scale)
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