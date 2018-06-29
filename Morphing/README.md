### MORPH MULTIPLE IMAGES
--------------------------------------------------------------------------

#### USAGE
Code uses OpenCV 3.3.1, numpy.
Download all files in the same directory.

For the images and stored points in this directory, run morph.py


If you want to select your own points, run get_points.py to select and store points and delaunay.py to store the delaunay triangles.

It is assumed that all images are of the same size and format is '.jpg'. If not, you will have to modify the codes accordingly.

#### 1) Run get_points.py to choose the points you want.
   
   Give the image name without the format as the command line argument. 
   
   For eg:  python .\get_points.py --image batman_affleck
   
   When the image opens, you can select and store points by clicking left mouse button on the point location.


#### 2) Run delaunay.py
   
   You will get a text file of the triangles calculated using Delaunay triangulation. To ensure that you get the same triangles every time, use any two images.
   
#### 3) Run morph.py to see the animation.
   
   You can change the sequence of images from the code.

--------------------------------------------------------------------------

#### WORKING

get_points.py uses a mouse callback function to store selected points plus four corners and the midpoint of the upper edge in a text file.

delaunay.py calculates the weighted average of the selected points of two images, and creates Delaunay triangles using averaged points as vertices. This ensures that same triangles are created for each image. This list of vertices is stored in a text file.

morph.py morphs individual triangular regions of the images using the text file created in delaunay.py. Then images are morphed using alpha blending. By varying the value of alpha, we can see the animation.
