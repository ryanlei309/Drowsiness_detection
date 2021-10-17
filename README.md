# Drowsiness_detection
Driving under drowsy condition is one of the main reason for car accident. Besides, since drivers sometimes have to drive with a tired body, so that we hope to use a computer vision system that can automatically detect driver drowsiness in a real-time video stream and then play an alarm if the driver appears to be drowsy. 

Meanwhile, during the pandemic, people work or study remotely more and more frequently. This computer vision can help people to judge whether they are work or study under a good condition.


# Dataset
The datasets used in our project is from Kaggle and provided ourselves.

We used 2,460 photos with different brightness and angle for training, including opened and closed eye, yawn and no-yawn. For validation data datasets, 432 photos were used.


# Methdology and Results
In our drowsiness detector case, we use the OpenCV module to launch the webcam to catch the eyes region.

After the webcam will take images from the live stream video, then our code function will input those images into our training model to identify whether the eyes are opened or closed. The following figure is our infrastructure of CNN.

![image](https://user-images.githubusercontent.com/52303625/137620545-c82c9afd-470b-40cf-a635-b8f5280cb76b.png)

Our best model has a combination of 3 layers, number of nodes [32, 32, 64], 100 epochs, 50% dropout, and achieving 95% accuracy on test data sets.

The longer time your eyes closed, the higher score you get. When the score achieve to certain point, a alarm goes off.

![image](https://user-images.githubusercontent.com/52303625/137620559-bf81a241-02cd-4559-bf9a-e8e5a5972033.png)


# Discussion and conclusion
For the datasets and training model we used, our training accuracy up to 98% with 100 epochs, and test dataset accuracy is nearly 95%. 

Nostrils may be detect as an eye on  webcam.


# Future work
Include more features to a neural network, such as nodding.

Further, discuss whether the cause of fatigue driving is related to the road section and t


# References
https://data-flair.training/blogs/python-project-driver-drowsiness-detection-system/

Datasets: https://www.kaggle.com/adinishad/driver-drowsiness-using-keras





