### TRACK MOTION OVER VIDEO FRAME AND RECOGNIZE AIR DRAWN DIGITS USING SVM
--------------------------------------------------------------------------

#### USAGE
Code uses OpenCV 3.3.1, numpy, scikit-learn, pickle

Download train_digits.py, common.py and digits.png in same directory.
Run train_digits.py to train and save SVM classifier using HOG descriptors. You can try different parameters for getting HOG descriptors.

Keep motion_tracker.py, img_proc.py and model_pickle.pkl (pretrained model) in same directory.
Next run motion_tracker.py, when Video stream appears, you can draw any digit from 0-9 in the air and see the predicted output in a second window. You can change motion detection parameters in motion_detection class from img_proc.py

Take care to keep background relatively still.

#### WORKING
This code uses a simple motion detector based on frame difference. When a motion is detected, its location is saved in an array. Each such location is shown on the frame by a pink square. The past 80 such locations are stored and displayed to give visual feedback of the path of your motion.

When no motion is detected for 20 consecutive video frames, the image created by the motion path (white motion path on a black background) is resized to the dimension of images in training database.

HOG descriptors are computed from this image and passed to the SVM classifier. The predicted output is displayed in a new window.
