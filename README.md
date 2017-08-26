# Face Recognition using DNN

*Free and open source face recognition with deep neural networks.*

This Git Repository is a collection of various papers and code on the face recognition system using **Python 2.7** **dlib 19.4.0** and **Skimage 0.9.3**.

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
There are many `.py` files in this repo, and they are used for differnt purposes.

The main file is `dlibopen.py` which is the face recognition software. There is a folder linked with this python file and i.e [images](/images) folder. The faces which needs to be recognized goes in this folder.

Once the images are there in the folder, we can start training on the persons images provided. We identify *128 points* on the face which is unique to all of them, and by importing 2 files which is an opensource shape detection and face recognition model provided by dlib.
```
shape_predictor_68_face_landmarks.dat
dlib_face_recognition_resnet_model_v1.dat
```