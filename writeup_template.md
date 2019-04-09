## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"
[project_video]: ./test_videos_output/project_video.gif "project_video"
[undistorted_1]: ./output_images/undistorted_1.png "undistorted"
[corrected_1]: ./output_images/correct_for_distortion1.png "corrected1"
[thresh_1]: ./output_images/gradient_thresh1.png "thresh1"
[static_trans1]: ./output_images/static_trans1.png "strans1"
[dynamic_trans1]: ./output_images/transform_array.png "dtrans1"
[poly1]: ./output_images/threshed1.png "poly1"
[curvature_formula]: ./output_images/curvature_formula.png "curvature_formula"
[warped_back1]: ./output_images/warp_back1.png "warpedback"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for camera calibration is located in the notebook "Camera Calibration.ipynb" in the folder notebooks/.  

The point of computing the camera matrix and the distortion coefficients is to represent the world in the most accurate possible. The **distortion coefficients** [k_1, k_2, p_1 p_2, k_3] are used to correct for tangential and radial distortion in the cameras. The **camera matrix** depends on the camera only, and includes info like focal length, and optical center of the camera. 

In order to compute the camera matrix and distortion coefficients, I took 20 photos of a checkerboard pattern in different orientations. For each photo, I used OpenCV's **findChessBoardCorners** functions to find 2D coordinate of where the coordinates are. For each of checkboards in 3D coordinates, we know that z = 0, and each corner is a fixed offset from another, so we build a 3D points template called **objp_template**. With the 2D points from **findChessBoardCorners** and 3D points **objp_template**, we can find a correspondence using OpenCV's **calibrateCamera** function to return the camera matrix and distortion coefficients. 

From the image below, you see that some of the curved lines on the edges are now straight. This means it corrected for radial distortion. 
![undistorted][undistorted_1] 


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstarte this step, here is a side by side view of one of the results:
![alt text][corrected_1]

To do this, I saved the **camera matrix** and **distortion coefficients** in a pickle file and loaded them. I supplied these coefficients to OpenCV's **undistort** function. The point of undistorting is to correct for distortion, because the pipeline for finding lane lines, curvature especially, is very sensitive to that. The code for this is section #2 in Pipeline.ipynb


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color, gradient, and direction thresholds to generate a binary image. This is under section #3 in Pipeline.ipynb. For color, I chose to use the color HSL space, and used the S component of it in particular from values (170, 255) to detect the lane lines. For gradient, I computed the x-gradient and y-gradient separately, and picked out gradient values between (30, 100). For direction, I calculated the gradient direction, and picked values (0.7, 1.3) to pick out vertical lane lines. 

![alt text][thresh_1]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
I experimented with two ways for the perspective transform: 1) a Static Way 2) Hough Line Finding Way

For the static way, I simply picked a trapezoid that seemed reasonable with points of the following and transformed it. The code and points are as follows:


```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

Static Transformation using Fixed Points:
![alt text][static_trans1]

I also tried another way where I found Hough Lines for the src points. As you can see the Hough lines detected more of the road. However, when we transformed to the destination, the lane line is much blurrier. Thus, the dynamic transformation seems to be not as robust.  
![alt text][dynamic_trans1]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

After I perspective transformed into the "Bird Eye View", I identified the lane-line pixels to fit the polynomial.

![alt text][poly1]

In my code, this is under "Pipeline.ipynb" section #4, in functions **fit_poly**, **find_lane_pixels**, **search_around_poly_with_curvature**, and **fit_polynomial_with_curvature**. In essence, 
1) we divide the bird eye's image into a left-half and a right-half for the left_lane and right_lane. 
2) Find the nonzero pixels in each half and give them a color(red/blue)
3) For **fit_polynomial_with_curvature**, start from scratch and use a sliding window to find th mean of each window to fit a polynomial
3)  For **search_around_poly_with_curvature**, we do not start from scratch, but instead parameters **left_fit**, **right_fit**, and **margin** are passed in so we  look for a polynomial fit based on a prior fit



#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code for ROC is also in "Pipeline.ipynb" section #4, in functions **search_around_poly_with_curvature**, **fit_polynomial_with_curvature**, **measure_curvature_real**
To calculate the curvature, we use the fromula

![alt text][curvature_formula]

We apply this formula but have to translate the value from pixels to meters in the real world. 
```python
 ym_per_pix = 30/img_shape[1] # meters per pixel in y dimension
 xm_per_pix = 3.7/img_shape[0] # meters per pixel in x dimension
```

The code for calculating the position of the vehicle with respect to center is in "Pipeline.ipynb" section #6 , in function **process_video**

To do this, I take the end points of the line to calculate the middle and compare against the image center. To calculate the end points of the line, the following python code is used 
```python
# Get Endpoint
pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
endPoints = np.array([pts_left[0][-1], pts_right[0][0]], dtype=np.float32)
endPointsTransformed = cv2.perspectiveTransform(np.array([endPoints]), Minv)
leftEndPoint, rightEndPoint = endPointsTransformed[0]
# Lane and Camera Center
laneCenter = (leftEndPoint[0] + rightEndPoint[0]) // 2
cameraCenter = img_size[0] // 2
# Calc Offset
offset = np.abs(laneCenter - cameraCenter) * xm_per_pix
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this reverse warping using the following python code that involves using the matrix **Minv** which warps points from **dst** to **src**. This **Minv** matrix is the inverse of the **M** matrix that is used to warp points from **src** to **dst**, which was used to create the bird's eye view. 

```python
pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))
# Draw the lane onto the warped blank image
cv2.fillPoly(color_warp, np.array([pts], 'int32'), (0,255, 0))
# Warp the blank back to original image space using inverse perspective matrix (Minv)
newwarp = cv2.warpPerspective(color_warp, Minv, img_size) 
```
What this code is doing is first getting the **pts** in the bird's eye view, filling a green polygon with the points, then warping the points back to the original lane. The results look like this:

![alt text][warped_back1]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./test_videos_output/project_video.mp4)


Or a Gif Here

![alt text][project_video]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
