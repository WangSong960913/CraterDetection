# CraterDetection
Crater detection algorithm based on deep learning and semantic segmentation
Our algorithm contains three main steps. 
Firstly, We need to generate experiment data.Our original moon DEM image can download in https://pan.baidu.com/s/1eSpBLrA-Upqr5qjf6__r8w.  and We random cliped the lunar DEM image to generate data. 
Secondly, using Simple-ResUNet detects crater edges in lunar images
Then, it uses template matching algorithm to compute the position and size of craters. 
Finally, we draw the images of recognized craters and record the location and radius of the craters.
##Dependencies
Python version 3.5+
Cartopy >= 0.14.2. Cartopy itself has a number of dependencies, including the GEOS and Proj.4.x libraries. (For Ubuntu systems, these can be installed through the libgeos++-dev and libproj-dev packages, respectively.)
* h5py >= 2.6.0
* Keras 1.2.2 
* Numpy >= 1.12
* OpenCV >= 3.2.0.6
* pandas >= 0.19.1
* Pillow >= 3.1.2
* PyTables >=3.4.2
* TensorFlow 0.10.0rc0, also tested with TensorFlow >= 1.0
