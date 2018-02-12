## Udacity Self Driving Car Nanodegree
## Vehicle Detection
## Project Writeup

---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/Image1.jpg
[image2]: ./output_images/image2.png
[image3]: ./output_images/trainGOOD_HOG_scale1_win32.png
[image4]: ./output_images/trainGOOD_HOG_scale1_win64.png
[image5]: ./output_images/trainGOOD_HOG_scale1_win128.png
[image6]: ./output_images/trainGOOD_HOG_scale1p5_win16.png
[image7]: ./output_images/trainGOOD_HOG_scale_1p5_win32.png
[image8]: ./output_images/trainGOOD_HOG_scale1p5_win64.png
[image9]: ./output_images/trainGOOD_HOG_scale1p5_win128.png
[image10]: ./output_images/trainGOOD_HOG_scale2_win64.png
[image11]: ./output_images/trainGOOD_HOG_scale2_win100.png
[image12]: ./output_images/trainGOOD_HOG_scale2p5_win64.png
[image13]: ./output_images/trainGOOD_HOG_scale3_win64.png
[image14]: ./output_images/image14.png
[image15]: ./output_images/Test_images.png
[image16]: ./output_images/Heatmap1.png
[image17]: ./output_images/Heatmap2.png
[image18]: ./output_images/Heatmap3.png
[image19]: ./output_images/Heatmap4.png
[image20]: ./output_images/BBox1.png
[image21]: ./output_images/BBox2.png
[video1]: ./output_images/project_video_final.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 56 through 69 of the file called `detect.py`, which references to the code contained in lines 45 through 98 of the file called `helper_functions.py`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and I discovered that the following combination yielded the highest accuracy rates on a consistent basis for the test data sets that I used to validate my classifier:

	color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
	orient = 8  # HOG orientations
	pix_per_cell = 8  # HOG pixels per cell
	cell_per_block = 2  # HOG cells per block
	hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM by first using color transformation (from RGB to the YCrCb color space) on a training image and then computed the binned color features, the color histograms and the HOG feature vector of that training image and concatenated them together into one feature vector (lines 45 through 98 of `helper_functions.py`).  I then applied the same process to all of the training images (both vehicles and non-vehicles images) in the data set.  Next, I normalized the data set using a StandardScaler before randomizing it and splitting it into training and testing sets (in 80:20 ratio) (lines 71 to 84 of `detect.py`) - which were then used to train and validate the classifier.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented the HOG sub-sampling window search technique that was taught in class into my code (lines ##). This technique is an efficient form of the sliding window method that allows us to only have to extract the Hog features once.  As described in the course material:

	"The find_cars only has to extract hog features once and then can be sub-sampled to get all of its overlaying windows. Each window is defined by a scaling factor where a scale of 1 would result in a window that's 8 x 8 cells then the overlap of each window is in terms of the cell distance. This means that a cells_per_step = 2 would result in a search window overlap of 75%. Its possible to run this same function multiple times for different scale values to generate multiple-scaled search windows."

I initutively set the cell_per_step to 1 (to maximize the number of overlapping windows) and tried many different combinations of scale values and window sizes to arrive at the optimal configurations.  My strategy is to find a combination that worked well with smaller car images and one that worked well with larger car images.  Some of my findings are displayed below:


![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]


Based on my results, I have chosen the configurations "scale=1, window size=128" to detect small car images and the configurations "scale=2, window size=100" to detect larger car images.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales and window sizes ("scale=1, window size=128" and "scale=2, window size=100") using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image14]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/project_video_final.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video via two different scans: one using a smaller scale value and one using a higher scale value.  I created two separate heatmaps from the two scans, thresholded the map created with the higher scale value (with a high threshold value to extract the larger vehicle images) and combined the two maps together before thresholding the combined map again to identify the vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the combined heatmap.  I then thresholded the blobs to determine which blobs correspond to a vehicle and which blobs correspond to a false positive.  I then constructed bounding boxes to cover the area of each blob that I have determined to be a vehicle.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are the original test images:
![alt text][image15]

### Here are the heatmaps of the six frames using a smaller scale value:
![alt text][image16]

### Here are the heatmaps of the six frames using a higher scale value (before thresholding):
![alt text][image17]

### Here are the heatmaps of the six frames using a higher scale value (after thresholding):
![alt text][image18]

### Here are the combined heatmaps of the six frames (after thresholding):
![alt text][image19]

### Here is the output of `scipy.ndimage.measurements.label()` on the combined heatmap from all six frames (before thresholding):
![alt text][image20]

### Here is the output of `scipy.ndimage.measurements.label()` on the combined heatmap from all six frames (after thresholding):
![alt text][image21]

A few interesting points to note from the above results:

- Only a small part of the black car is detected in the heatmap of Image8 created with the smaller scale value (i.e. large vehicles are not properly detected)
- The white car is invisible in the heatmap of Image3 created with the higher scale value (i.e. small vehicles are not properly detected)
- The two issues above are fixed when we created the combined heatmap 
- Without thresholding the blobs, the output of label will include false positives if we draw a bounding box on all detected objects
- After thresholding the blobs, the output of label will no longer include any false positives since we have not drawn bounding boxes on those false positives

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Initially, I chose a scale value of 1.5 and window size of 65 to create the heat map, which worked well and had relatively few false positives when I ran the video through my pipeline.  But this configuration didn't work well with vehicles that were further away.  So I decided to switch to a scale value of 1 and window size of 128 to create the heat map, which worked well with smaller vehicle images but it didn't work well with larger vehicle images (it only detected a portion of the larger vehicle images).  After further experimentation, I decided to scan each frame twice: one with a smaller scale value and one with a larger scale value - this helped my code to accurately detect both small vehicle images and larger vehicle images.  To help minimize false positives, I decided to threshold the blobs detected in the labeled heatmaps to locate the vehicles, but drawing bounding boxes on the end result directly is not ideal as I used a high threshold value, which was good in weeding out the false positives but it wasn't good in capturing the entirety of the vehicles since anything apart from the "hotest" pixel values of the vehicles would be thresholded out.  In the end, I decided to first extract the vehicle label values by thresholding out the false positives in the labeled heatmap and then used the vehicle label values to draw the bounding boxes on the original labeled heatmap (not the thresholded labeled heatmap), which allowed me to capture a greater portion of the vehicles in bounding boxes. 

The biggest issue with my pipeline is efficiency - it takes a long time to render the output video.  The 2 step scanning process and multiple thresholding approach are good in acurately identifying the vehicles but it is not that efficient.  If I have more time to work on the project, I would conduct more experimentations to see how I could optimize my steps to make my pipeline more efficient.

Also, training with 8000 or so images is probably not enough and might cause my classifier to mis-identify vehicle and non-vehicle images.  To make my classifier more robust, I should use more training images during training.
