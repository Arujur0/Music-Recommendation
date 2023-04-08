# Music-Recommendation
A music recommendation system derived from facial emotions. Consists of a facial emotion detector constructed using a neural network and a recommendation system built using filtering methods.

## Introduction
This project is a facial emotion recognition project that uses deep learning techniques to recognize human emotions from facial expressions and use those emotions to generate accurate song recommendations. This project is split into two phases:
1. Facial Emotion Recognition: Accurately identify the seven basic emotions such as happiness, sadness, anger, fear, disgust, and surprise and neutral (calm) from facial images in real time.
2. Music Recommendation: Recommend songs from the Spotify API using the results of the facial emotion detection system.

## Dataset
The dataset used for the facial emotion Recognition task is the [FER2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) dataset. It consists of 35,887 grayscale images of size 48x48 pixels, with each image labeled with one of seven emotions: anger, disgust, fear, happiness, sadness, surprise, and neutral. 
For the task of music recommendation, playlists were scrapped from the [Spotify Web-API](https://developer.spotify.com/documentation/web-api/).

## Architecture
For the task of Facial Emotion Recongition, a simple Convolutional Neural Network is constructed. The convolutional Neural Networks consists of six convolutional blocks. Each block is made up of a sequence of a convolutional layer, followed by Batchnorm and ReLU activation function. There is also a three layer MLP that makes up the classification section of the neural network. The architecture is shown in ![Model Summary](layers.png)


## Results and Discussion
Facial Emotion Recognition

![Happy Class](test_1.png)

![Neutral Class](neutrals.png)
