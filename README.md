# FacialLandmarkDetection


## In Brief, ##

The main objective of this repo is to predict and localize the keypoint/landmarks positions on face images. Facial keypoint detection is used for several application such as
  * Face tracking in video/image
  * Biometrics (Security for phones)
  * Face Recognition
  * Facial Expression Analysis
  * Advanced Behavioral Data Analysis
  * Detection of Dysmorphic Facial Signs for Medical Diagnosis
  * Deep Fakes


Facial landmarks vary greatly from one individual to another, there is lots of variations because of the 3-D pose, face size, weight, perspective so that it is really challenging taks. Moreover, computer vision research has come a long way in addressing these difficulties, but there remain many opportunities for improvement [1]. In this repository, I implemented both custom convolutional neural network and ResNet152, VGG19 to detect landmarks on YouTube Faces Dataset (link is provided below). It is a dataset that contains 3,425 face videos designed for studying the problem of unconstrained face recognition in videos. These videos have been fed through processing steps and turned into sets of image frames containing one face and the associated keypoints [2]. 

As a image augmentation, the following techniques are utilized.
* RandomCrop
* ColorJitter
* RandomGrayScale
* GaussianBlur

As a preprocessing, the following methods are used.
* Resizing to 3 x 224 x 224
* Conversion of PyTorch Tensor
* Custom Normalization

As a feature extractor, I implemented custom CNN with the following convolutional block with 6 blocks,

* nn.Conv2d(in_channel, out_channel, 3) -> nn.BatchNorm2d(out_channel) -> nn.ReLU() -> nn.MaxPool2d(2,2) -> nn.Dropout2d(p)

Then, extracted features follows the fully connected blocks with the following block type,

* nn.Linear(in_ftrs,out_ftrs) -> nn.BatchNorm1d(num_features=out_ftrs) -> nn.ReLU() -> nn.Dropout(p)

Hence, the following architecture is constructed to detect the keypoints on face images.

Model(
  (model): Sequential(
    (0): Sequential(
      (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1))
      (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (4): Dropout2d(p=0.1, inplace=False)
    )
    (1): Sequential(
      (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (4): Dropout2d(p=0.15, inplace=False)
    )
    (2): Sequential(
      (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
      (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (4): Dropout2d(p=0.2, inplace=False)
    )
    (3): Sequential(
      (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
      (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (4): Dropout2d(p=0.25, inplace=False)
    )
    (4): Sequential(
      (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1))
      (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (4): Dropout2d(p=0.3, inplace=False)
    )
    (5): Sequential(
      (0): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1))
      (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (4): Dropout2d(p=0.35, inplace=False)
    )
    (6): Flatten(start_dim=1, end_dim=-1)
    (7): Sequential(
      (0): Linear(in_features=1024, out_features=512, bias=True)
      (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): Dropout(p=0.4, inplace=False)
    )
    (8): Sequential(
      (0): Linear(in_features=512, out_features=256, bias=True)
      (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): Dropout(p=0.4, inplace=False)
    )
    (9): Linear(in_features=256, out_features=136, bias=True)
  )
)


- - - -


 * Keywords
    * Convolutional Language Model for Image Captioning
    * Deep Learning for Vision & Language Translation Models
    * Transfer Learning
    * Data Augmentation
    * Parallel Distibuted Processing
    * Attention model and Teacher Forcer Algorithm
    * AlexNet, VGG-Net, ResNet, DenseNet and SquezeeNet
    * Long-Short Term Memory(LSTM) and Gated Recurrent Unit (GRU)
    * Global Vector for Word Representation(GloVe) 
    * Beam and Greedy search
    * BLEU Scores and METEOR

- - - -

Here are some samples from our Vision & Language Models:
 

![DetectedLandmarks2](https://user-images.githubusercontent.com/53329652/105414611-c6582c00-5c48-11eb-9ccd-249e9b8ea5a8.png)
![DetectedLandmarks](https://user-images.githubusercontent.com/53329652/105414617-c7895900-5c48-11eb-961f-652697cbdf0f.png)


Dataset Link: https://www.cs.tau.ac.il/~wolf/ytfaces/
