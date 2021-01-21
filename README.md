# Facial Landmark Detection


## In Brief, ##

The main objective of this repo is to predict and localize the keypoint/landmark positions on face images. Facial keypoint detection is used for several application such as
  * Face tracking in video/image
  * Biometrics (Security for phones)
  * Face Recognition
  * Facial Expression Analysis
  * Advanced Behavioral Data Analysis
  * Detection of Dysmorphic Facial Signs for Medical Diagnosis
  * Deep Fakes

Facial landmarks vary greatly from one individual to another, there is lots of variations because of the 3-D pose, face size, body weight, perspective and much more so that it is really challenging task in the context of computer vision. Recently, computer vision research has come a long way in addressing these difficulties, but there remain many opportunities for improvement [1]. In this repository, I implemented both custom  and pretrained (ResNet152, VGG19) convolutional neural network to detect landmarks on YouTube Faces Dataset (link is provided below). It is a dataset that contains 3,425 face videos designed for studying the problem of unconstrained face recognition in videos. These videos have been fed through processing steps and turned into sets of image frames containing one face and the associated keypoints [2]. 

The packages used for this repo is provided below.

| Packages      |   Versions    |
| ------------- | ------------- |
| PyTorch       | 1.7.0+cu101   | 
| Torchvision   | 0.8.1+cu101   |
| NumPy         | 1.19.5        |
| OpenCV        | 4.1.2         |
| Pandas        | 1.1.5         |
| Pillow        | 7.0.0         |
| Scikit-Image  | 0.16.2        |


----
Here are some sample landmark localizations from my custom facial landmark detector:
 
![DetectedLandmarks2](https://user-images.githubusercontent.com/53329652/105414611-c6582c00-5c48-11eb-9ccd-249e9b8ea5a8.png)
![DetectedLandmarks](https://user-images.githubusercontent.com/53329652/105414617-c7895900-5c48-11eb-961f-652697cbdf0f.png)


As a image augmentation, the following techniques are utilized.
* RandomCrop
* ColorJitter
* RandomGrayScale
* GaussianBlur

As a preprocessing, the following methods are used.
* Resizing to 3 x 224 x 224 convention
* Normalization

As a feature extractor, I implemented custom CNN with the following convolutional block (total number of 6 blocks),

* nn.Conv2d(in_channel, out_channel, 3) -> nn.BatchNorm2d(out_channel) -> nn.ReLU() -> nn.MaxPool2d(2,2) -> nn.Dropout2d(p)

Then, extracted features follows the fully connected blocks with the following block type (total number of 2 blocks),

* nn.Linear(in_ftrs,out_ftrs) -> nn.BatchNorm1d(out_ftrs) -> nn.ReLU() -> nn.Dropout(p)

Lastly, linear layer is added at the end of the fully connected block to predict continious keypoint values.

In my custom model, there are 6,984,840 (≈ 7 million) trainable parameters that should be optimized by Adam optimizer with the scheduled learning rates (Initial learning rate is  0.001).

SmoothL1Loss is used a a loss function, uses a squared term if the absolute element-wise error falls below beta and an L1 term otherwise. It is less sensitive to outliers than the MSELoss and in some cases prevents exploding gradients (e.g. see Fast R-CNN paper by Ross Girshick) and also known as the Huber loss [3].

Then, adaptive learning rate scheduler is used (ReduceLROnPlateau) to monitor the validation loss, learning rate is decreased when model stops improvements per  5 epoch).

Finally, the OpenCV packages Face CascadeClassifier is utilized to get bounding box of input image, and feeded to the network to test our model on unseed natural data.

Also, feature extraction experiments conducted by using ResNet152 and VGG19 to compare findings from custom model.

- - - -

Dataset Link: https://www.cs.tau.ac.il/~wolf/ytfaces/


Reference Links

[1] [https://www.kaggle.com/c/facial-keypoints-detection

[2] https://towardsdatascience.com/facial-keypoint-detection-using-cnn-pytorch-2f7099bf0347

[3] https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html

