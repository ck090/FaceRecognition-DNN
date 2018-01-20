# Face Recognition using DNN

*Free and open source face recognition with deep neural networks.*

This Git Repository is a collection of various papers and code on the face recognition system using **Python 2.7**, **dlib 19.4.0** and **Skimage 0.9.3**.

The .pdf files in this repo are some of the earliest and the fundamental papers on this topic. It also includes the Research paper from Google's *FaceNet* and Taigman's *DeepFace*.

# Dependencies
* numpy
* opencv 2.4.8
* matplotlib
* dlib 19.4.0
* os
* skimage 0.9.3
* scipy 0.13.3
Use pip or easy_install to install any missing dependencies.

# Usage
There are many `.py` files in this repo, and they are used for different purposes.

The main file is `dlibopen.py` which is the face recognition software. There is a folder linked with this python file and i.e [images](/images) folder. The faces which needs to be recognized goes in this folder.

Once the images are there in the folder, we can start training on the persons images provided. We identify *128 points* on the face which is unique to all of them, and by importing 2 files which is an opensource shape detection and face recognition model provided by dlib.
```
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
```

Then there are several functions which are used from drawing rectangles on the face frame... many faces can be deteceted at the same time (Also the names found are also put up on the screen to give a better UI experience), to identifying the *facial landmarks* within the image. 

We have set the threshold to 0.6 which is the mid-way between recognizing false negatives and false positives.

Also the frames are taken in `modulus 2` fashion to improve on the efficieny of running the software. (i.e every other frame is chosen). It's upto the users to change it if they have a gpu environment system.

# How to run
1. Git clone or download this Repository (don't forget to STAR it!! 😀)
2. Store the images you want to recognize in the [images](/images) folder. (one picture per person is enough).
3. Make sure you have all the dependencies installed.
4. Run the `dlibopen.py` from within the folder.
5. It should pretty much start running the software.

Here's a picture of it recognizing Chad from RHCP.
<img width="621" alt="chad" src="https://user-images.githubusercontent.com/12717969/29740532-3bd14a0a-8a76-11e7-9067-ae50c79f2dde.png">