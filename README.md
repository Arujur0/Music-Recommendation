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
For the task of Facial Emotion Recongition, a simple Convolutional Neural Network is constructed. The convolutional Neural Networks consists of six convolutional blocks. Each block is made up of a sequence of a convolutional layer, followed by Batchnorm and ReLU activation function. There is also a three layer MLP that makes up the classification section of the neural network. The architecture is shown in the figure below: ![Model Summary](layers.png)


## Results and Discussion
Facial Emotion Recognition
The results of the FER system are a class label that best represents the user's current emotional state. An interactive User Interface is generated using Flask, basic HTML/CSS and JavaScript. The UI presents an option for user's to give access to their camera and for the model to detect the bounding box of their face and generate their emotional state frame by frame in real time. It also allows user's to decide which emotion they would like to recieve recommendations for. Sample predictions made by the model are below:
![Happy Class](test_1.png)

![Neutral Class](neutrals.png)

The model was evaluated on metrics such as Accuracy, Precision and Recall. A heatmap of the confusion matrix was also generated for better understanding of what the model is able to predict efficiently. The FER model achieves an accuracy of 62.17%, a precision of 62.2% and a recall of 61.15%. The heatmap is shown below:
![Confusion Matrix](conf_mat.png)
As the number of samples for the disgusted class label is very low and because it is very difficult to sample music that accurately represents songs that induce a complex emotion like disgust, the disgust class was combined with anger to form a single class.

