import matplotlib.pyplot as plt
import traceback
import numpy as np
import cv2
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys
import os
import pickle
from helper_functions import *
import glob
import time
from moviepy.editor import VideoFileClip

# Variables used for model training and car finding (via the window technique)
color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)  # Spatial binning dimensions
hist_bins = 32  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off

### PART ONE: TRAIN A CLASSIFIER
### In this part of the code, we will train a classifier that can detect vehicles
### using the labelled training data from the GTI vehicle image database and the
### KITTI vision benchmark suite.  Once the classifier has been trained, we will
### save the classifier (together with the relevant parameters) to a pickle file
### for future uses to avoid having to retrain the classifier every single time
### we run the video pipeline.

filename = "train.p"
file_list = os.listdir(os.getcwd())

# Build a classifier if pickle file not found
if filename not in file_list:


    # The training data from the GTI vehicle image database
    # and the KITTI vision benchmark suite are stored in separate
    # folders and thus can be differentiated here by their storage location
    images = glob.glob('train_images\\**\\*.png', recursive=True)
    cars = []
    notcars = []
    for image in images:
        if 'non-vehicles' in image:
            notcars.append(image)
        else:
            cars.append(image)

    # Extract the color space, spatial binning and HOG features of the training data
    # in a concatenated form
    car_features = extract_features(cars, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)

    # Create an array stack of feature vectors
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
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

    # Save the classifier and relevant parameters for future use
    dist_pickle = {}
    dist_pickle["svc"] = svc
    dist_pickle["scaler"] = X_scaler
    dist_pickle["orient"] = orient
    dist_pickle["pix_per_cell"] = pix_per_cell
    dist_pickle["cell_per_block"] = cell_per_block
    dist_pickle["spatial_size"] = spatial_size
    dist_pickle["hist_bins"] = hist_bins
    with open(filename, "wb") as f:
        pickle.dump(dist_pickle, f)
else:
    # Pickle file found, so no training is required.
    # Retrieve classifier from file instead
    try:
        with open(filename, "rb") as f:
            dist_pickle = pickle.load(f)
    except (AttributeError, EOFError, ImportError, IndexError) as e:
        print(traceback.format_exc(e))
        pass
    except Exception as e:
        print(traceback.format_exc(e))
        sys.exit()

    # Load pickie file data into the local variables
    svc = dist_pickle["svc"]
    X_scaler = dist_pickle["scaler"]
    orient = dist_pickle["orient"]
    pix_per_cell = dist_pickle["pix_per_cell"]
    cell_per_block = dist_pickle["cell_per_block"]
    spatial_size = dist_pickle["spatial_size"]
    hist_bins = dist_pickle["hist_bins"]

### PART TWO: TEST THE CLASSIFIER ON STATIC IMAGES
### Once the classifier has been trained, I wrote two test functions (hog_test and heat_test)
### to test the accuracy and effectiveness of both the classifier and the window searching technique
### in identifying vehicles on the static images that was provided with the project files.
### Different parameter values can be passed into these two functions allowing me to continuously
### tweak the relevant parameter values in order to find the optimal configurations.

# This test function runs the hog sub-sampling window search technique with the classifier
# on all 6 static images and displays the results on one page for easy comparison
def hog_test(plt, ystart, ystop, scale, window_size, cells_per_step):
    # Load all static images into a list
    image_names = ["test1.jpg", "test2.jpg", "test3.jpg", "test4.jpg", "test5.jpg", "test6.jpg"]
    file_names = ["test_images/" + name for name in image_names]
    imgs = list(map(lambda img: cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB), file_names))

    out_imgs = []

    for img in imgs:
        # The hog sub-sampling window search technique is performed on every single image
        matched_boxes = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                        hist_bins, color_space, window_size=window_size)

        out_imgs.append(draw_boxes(img, matched_boxes))

    title = ["Scale = " + str(scale), "Window Size = " + str(window_size), "Cells Per Step = " + str(cells_per_step)]

    # Plot the results on one page
    plt = plotimages(plt, out_imgs, captions=image_names, hspace=0.3, nrows=3, ncols=2, title=title)
    plt.show()

# This test function runs the hog sub-sampling window search technique with the classifier
# followed by the heatmap and labelling code provided in the tutorial on all 6 static images
# and displays the results on one page for easy comparison
def heat_test(plt, ystart, ystop, scale_1 = 1, window_size_1 = 128, scale_2=2, window_size_2 = 100, cells_per_step = 1,
              threshold_2 = 8, combined_threshold = 1, correct_label_threshold = 6):
    # Load all static images into a list
    image_names = ["test1.jpg", "test3.jpg", "test4.jpg", "test5.jpg", "test6.jpg", "test7.jpg"]
    file_names = ["test_images/" + name for name in image_names]
    imgs = list(map(lambda img: cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB), file_names))

    out_imgs = []
    for image in imgs:
        # Create a first heat map with a smaller window
        # The hog sub-sampling window search technique is performed on every single image
        matched_boxes_1 = find_cars(image, ystart, ystop, scale_1, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                    spatial_size, hist_bins, color_space, window_size=window_size_1)

        # Initialize heat map with all pixels set to 0
        heat = np.zeros_like(image[:, :, 0]).astype(np.float)

        # Add heat to each box in box list
        heat = add_heat(heat, matched_boxes_1)

        # Create a second heat map with a larger window
        # The hog sub-sampling window search technique is performed on every single image
        matched_boxes_2 = find_cars(image, ystart, ystop, scale_2, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                    spatial_size, hist_bins, color_space, window_size=window_size_2)

        # Initialize heat map with all pixels set to 0
        heat_2 = np.zeros_like(image[:, :, 0]).astype(np.float)

        # Add heat to each box in box list
        heat_2 = add_heat(heat_2, matched_boxes_2)

        # Apply threshold to help remove false positives
        heat_2 = apply_threshold(heat_2, threshold_2)

        # Sum the two heat maps together
        heat += heat_2

        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, combined_threshold)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)

        # Identify the vehicle labels by thresholding the heat map
        correct_labels = set(labels[0][heat > correct_label_threshold])

        out_imgs.append(draw_correct_bboxes(np.copy(image), labels, correct_labels))

    title = ["Scale 1 = " + str(scale_1), "Window Size 1 = " + str(window_size_1), "Scale 2 = " + str(scale_2), "Window Size 2 = " + str(window_size_2), "Cells Per Step = " + str(cells_per_step)]

    # Plot the results on one page
    plt = plotimages(plt, out_imgs, captions=image_names, hspace=0.3, nrows=3, ncols=2, title=title, cmap='hot')
    plt.show()

# Running list of heat maps
heatlist = []


def process_image(image):

    ystart = 350
    ystop = 656
    # Parameters for first heat map
    scale_1 = 1
    window_size_1 = 128
    # Parameters for second heat map
    scale_2 = 2
    window_size_2 = 100
    threshold_2 = 8
    # Threshold for combined heat map
    combined_threshold = 1
    correct_label_threshold = 6

    # Create a first heat map with a smaller window
    # The hog sub-sampling window search technique is performed on every single image
    matched_boxes_1 = find_cars(image, ystart, ystop, scale_1, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                              spatial_size, hist_bins, color_space, window_size=window_size_1)

    # Initialize heat map with all pixels set to 0
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat, matched_boxes_1)

    # Create a second heat map with a larger window
    # The hog sub-sampling window search technique is performed on every single image
    matched_boxes_2 = find_cars(image, ystart, ystop, scale_2, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                              spatial_size, hist_bins, color_space, window_size=window_size_2)

    # Initialize heat map with all pixels set to 0
    heat_2 = np.zeros_like(image[:, :, 0]).astype(np.float)

    # Add heat to each box in box list
    heat_2 = add_heat(heat_2, matched_boxes_2)

    # Apply threshold to help remove false positives
    heat_2 = apply_threshold(heat_2, threshold_2)

    # Sum the two heat maps together
    heat += heat_2

    # Add heat map to running list
    heatlist.append(heat)

    # Trim the list if it exceeds 5 heat maps
    if len(heatlist) > 5:
        heatlist.pop(0)

    if len(heatlist) > 1:
        # Calculate the mean values of the running list of heat maps
        heat = cal_heat_mean(heatlist)
    else:
        # Just use the latest entry in the list
        heat = heatlist[-1]

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, combined_threshold)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)

    # Identify the vehicle labels by thresholding the heat map
    correct_labels = set(labels[0][heat > correct_label_threshold])

    return draw_correct_bboxes(np.copy(image), labels, correct_labels)

# Use code below for testing
# np.set_printoptions(threshold='nan')
# heat_test(plt=plt, ystart=350, ystop=656, cells_per_step=1)
# hog_test(plt=plt, ystart=350, ystop=656, scale=1, window_size=64, cells_per_step=1)

# Use code below for rendering videos
output = 'output_images/project_video_test.mp4'
clip1 = VideoFileClip("project_video.mp4").subclip(27,28)
clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
clip.write_videofile(output, audio=False)