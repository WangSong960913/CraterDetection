# CraterDetection
Crater detection algorithm based on deep learning and semantic segmentation
Our algorithm contains three main steps. 
Firstly, Generate data.Our original moon DEM image canWe cliped the lunar DEM image to 256x256. 
Firstly, Simple-ResUNet detects crater edges in lunar images using the convolution neural network. Then, it uses template matching algorithm to compute the position and size of craters. Finally, we draw the images of recognized craters and record the location and radius of the craters.
