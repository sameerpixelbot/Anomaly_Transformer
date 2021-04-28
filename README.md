# Anomaly Detection

> **Detecting anomalies in public area using human action recognition**

## Note 

This is our Final year project in BTech at [NIT Warangal](https://nitw.ac.in/main/).  
We are a team of three members [N Sai Shashank](https://www.linkedin.com/in/sai-shashank-n-370b6b155/), [M Sameer Ahmed](https://www.linkedin.com/in/sameer-ahmed-802195195/), [K Praneeth Reddy](https://www.linkedin.com/in/praneeth-kunduru-69797614b/). Here is the [paper]() submitted at [IEEE International Conference on Automatic Face and Gesture Recognition 2021](http://iab-rubric.org/fg2021/#:~:text=FG%202021%20IEEE%20International%20Conference%20on%20Automatic%20Face,and%20video-based%20face%2C%20gesture%2C%20and%20body%20movement%20recognition.).

## Introduction 

Due to the recent events of Covid-19, It has become very important to maintain good health and fitness. It has also become very important to monitor public areas to detect any sorts of symptoms related to any type of Disease, not just symptoms regarding diseases but also for there physical well being. So the project attempts to solve the above needs. In an attempt to solve the problem in a novel way and looking at the recent performance of models like GPT-3 we decided to make a transformer based model in order to solve the problem.

## Dataset 

For the data we wanted video files so we decided to go with [Rose labs Dataset](http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp). It has 120 classes and 114,480 samples in total. Due to computation limitation we decided to only predict 7 classes.  
These 7 classes are :  

1. Sit down
2. Stand up
3. Jump up
4. Sneeze/Cough
5. Staggering
6. Fall down
7. Nausea/Vomiting

All the input videos consisted of a human doing the above action.

## Model

People have attempted to use transformer based models in the past. But the training process on direct images or video data was quite expensive and time taking. Even when there were attempts to lower the resolution of images (like 16x16) the model was quite difficult to train. In order to solve this problem we have taught to extract key points of the human and send this key point co-ordinates(Joints of a body) of human body for training. Doing this gave excellent results!!! 😎 (results will be discussed in the next sections). So inorder to extract key points we used [HR-Net](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation) as it is the best model for our application.  

The Common problem faced by key point detection models is voids in key points due to loss of resolution in its pipeline, Occultation and incorrect key points detected varying scale of humans in input data. These problems causes a lot of difficulty in recognizing human action or pose. HR-Net solves all these problems by maintaining high resolution throughout its pipeline, trained on data that accounts for occultation and trained on scale aware data. So when extracting key points we take every 5th frame of the video as the in between frames are almost similar and we take a sequence of such 15 frames.  

Transformer based models are best known for sequence to sequence translation. So when the input has a sequential dependency like text sentences it performs the best in such situation. As human actions are also a sequence of poses which depend on each other transformer is a good model for this application. So we use a stack of Encoder layers of transformer which help us extract contextual embeddings of the given input tensor. Each encoder layer has multi attention heads and a feed forward layer. These attention heads help extracting the most important features from the input and passes this down to the next encoder layer. This process makes the embeddings of very high quality based on the input embeddings. These embedding help in classifying more effectively thus giving exceptional results. So after all the encoder layer we do a SoftMax classification to classify the input.

## Training

Once we extract the key point embeddings of each video we form an input tensor of shape 15x34 where 15 corresponds to the frames and 34 is the co-ordinates, HR-Net outputs 17 key points so 17*2=34 co-ordinates. The output is a tensor of shape 1x7 here it is in one hot encoding. The model was then trained on colab.

**So once trained on the data the model gave an accuracy of 95% on training data. Looking at this high performance we trained for up to 60 classes then reached an accuracy of 94%**

## Testing

**Our test data consisted of 280 videos and 7 classes. The first model gave an accuracy of 87%.  
The second model on the same testing data gave and accuracy of 63%**.

## Technologies used

- Python
- Pandas, Tensorflow, Keras.

## Future scope

- Trying to do predictions for up to 120 classes
- Real time predictions