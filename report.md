**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/sliding_windows01.jpg
[image4]: ./output_images/sliding_window.jpg
[image5]: ./output_images/bboxes_and_heat.png
[image6]: ./output_images/labels_map.png
[image7]: ./output_images/output_bboxes.png
[image8]: ./output_images/sliding_windows02.jpg
[image9]: ./output_images/sliding_windows03.jpg
[image10]: ./output_images/sliding_windows01.jpg
[image11]: ./output_images/HOG_example_1.png
[video1]: ./project_video.mp4


### Histogram of Oriented Gradients (HOG)

#### 1. Extracted HOG features from the training images.

The code extract HOG features defined in `get_hog_features` function in ines 21 through 38 of the file called `utils.py`, which just wrraper for `skimage.feature.hog` function. The function returns  features vector and also can return visualization of the HOG when visualise is set to True.

I started by loading random sample from each class, the code cell 7. below is an example of  the images:

![alt text][image1]

I then start experimenting with different color space and  orientations, pixels per cell and cells per block. and for each iteration I plotted HOG so I can see how the changes in the paramters impact HOG.

The code  in `Vehicle-Detection.ipynb` notebook cell 8. where the code first convert the images to the required color space and then calls `get_hog_features` to extract Histogram of Oriented Gradient features.


Below is an example for images from different class using `YCrCb` color space, 15 orientations, 8 pixels per cells and 2 cells per block:

![alt text][image2]

#### 2. Choice of HOG parameters.

To choice HOG parameters I decided to train the classifier with different combinations of the parameters, and then I appied the model on the test images to see how the classifier perform on unseen data.

At the end I found the model works better with below values:

- Channel 0 from `YCrCb` color space.
- 8 pixel/cell
- 2 cell/block
- 15 orientation.

#### 3. Training a classifier using your selected HOG features.

I trained a feed forward neural network instead of linear SVM using only HOG features extracted from dataset provided by Udacity. The model consist of four fully connected layers (1024/512/64/1) with Relu activation in all layer except the last output leyer wehere I used sigmoid function. The code for the model in line 8 though 15 in `model.py`.

I start by load the images line 19 through 21 in `train.py`  and extracting features line 55 through 62 in `data.py`. I then scaled the extracted features and split them up into traning, test and validation sets in line 70 through 86 in `data.py`.

After that I trained and saved the model in line 39 though 45.

You can train the model by running `train.py` script as showin in below example

~~~~
python ./train.py --model=model  --color_space=YCrCb --hist_bins=32  --orient=15  --pix_per_cell=8 --cell_per_block=2 --spatial_size=32  --hog_channel=ALL --no-spatial_feat --no-hist_feat --epochs=2
~~~~



#### Sliding Window Search

I decided to split the image into four regions with different scale and overlap to reduce the number of boxes the code in line 179 through 183 in `pipeline.py` which go through predefined regins and then call `find_cars` function line 163 through 170 in `pipeline.py`.

`find_cars` function calls `slide_window` function line 122 through 161 in `pipeline.py` to build list of boxes, and then calls `search_windows` line 69 through 87 in a `pipeline.py` to classify content of the boxes and filter out the boxes with none cars.

I select the boundries and scale ofthe regins empirically using below method:
1. Split selected region in the test inages using into segment using selected scale/overlap.
2. resize segement to 64 x 64 images and run them through the trained tclassifier.
3. Count number of false positive.
4. Select the configuartion with minmum false positive.

The code in notebook `Vehicle-Detection` cell 11 

And below is the result

1. X-axis [450 - End of image] / Y-axis [400 - 480] / size (85, 80) / 75% overlap
![alt text][image3]
2. X-axis [350 - End of image] / Y-axis [380 - 530] / size (130, 130) / 75% overlap
![alt text][image8]
3. X-axis [350 - End of image] / Y-axis [410 - 530] / size (95, 95) / 50% overlap
![alt text][image9]
4. X-axis [200 - End of image] / Y-axis [400 - 680] / size (230, 100) / 10% overlap
![alt text][image10]

Ultimately I searched on four scales using YCrCb 1-channel HOG features only, which provided a nice result.  Here are some example images:

![alt text][image4]

I imporoved classifier performnce by adding Dropout regularization and tunnied network hyper  parameters (Learning rate/batch size/ epochs). And the result 
was 0.992 accuracy on the validtion set and 0.990 in the test set.

I did trained the model  using spatially binned color and histograms of color along with HOG, but I didn't see visible improvement so I decided to use only HOG.

---

### Video Implementation


Here's my video result

[![ADVANCE LANE DETECTION](https://img.youtube.com/vi/a1rPXk_x8S8/0.jpg)](https://youtu.be/a1rPXk_x8S8 "Vehicle Detection")

#### 2. Filter for false positives and some method for combining overlapping bounding boxes.

I used the positive detected boxes to create heatmap which mark all the positive position. I then applied predefined threshold to combin overlapping bounding boxes and remove the false positives. The code in line 90 through 97 in file `pipeline.py`.

Then I used `scipy.ndimage.measurements.label()` to identfiy individual blobs in the heatmap, which then used to construct bounding box  in the frame to cover detected blob which assumed to be a vehicle. The code in line 99 through 112 in file `pipeline.py`.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

Here are six frames and their corresponding heatmaps:

![alt text][image5]

Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]

---

### Discussion

I started by putting togther pipeline skeleton so I can identify different components, and to see how is everything fit togther. The second stage for was to focuse on the main two components, classifier to filter out the non-cars images and  algorithm to slice the images to small pieces (Sliding window).

I started with classifier where I train different type (Liner SVM / None-liner SVM / Neural network) using diiferent feature configurtions, so I can chose the model which has less false positive and can be trained in reusable amount of time. And what I found is simple feed frowrad neural network has less false positive then SVM classifier (Liner and Non-liner) and dones't take long to train (About 80 sconds).

I then start looking at which sliding windows implementation I should use, where one use predefined scale to scale down region of the image and then slide fix sized window like one implementated in `find_car`. And other one which can slide different size window in the same region.

The found the first  implementattion hard to tune, so I decided to the second implementation.

At this point what left is to see how to eliminate the false positive. For that I decied to use heatmap and `scipy.ndimage.measurements.label()`.

The current pipeline is far from prefect, I still see false positive and most case the box either smaller then the object or big. The false positive could be handling by tracking boxes detected in the last n frames and use it to eliminate false positive, or use better classifier.

For the big size  we could try and predict center for number of overlapped boxes and then use it to reject any point with distance from center higher then given threshold, And small size we could have minimum size.

We could also use deep learning approche instead of hand crafted features simple CNN or End-to-End trained model like YOLO or SSD, which predict class object and box from the image directly. 


