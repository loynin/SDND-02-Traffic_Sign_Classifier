# Traffic Sign Recognition

  It is easy for human naked eye to understand and to identify the traffic signs. On the other hand, for the self-driving car it is not easy. Therefore, the traffic sign recognition is important in the field of self-driving car. In order for self-driving car to understand the traffic condition and environment around it, the self-driving car must understand and recognize the traffic signs that post on the road or anywhere that the car navigate to.  If the self-driving car could not recognize the traffic signs, it would be unsafe and very dangerous and the self-driving car is not applicable on the road and in the real world. 
This project is about the traffic sign classification in helping self-driving car to identify the traffic signs. This project is based on machine learning and CNN. Start with preparing the training data and building the model to testing the real traffic sign, this project is aim to make sure the applicable of self-driving car in the real world. 

### Build a Traffic Sign Recognition Project
The steps of this project are the following:
- Load the data set 
- Explore, summarize and visualize the data set
- Design, train and test a model architecture
- Use the model to make predictions on new images
- Analyze the softmax probabilities of the new images
- Summarize the results with a written report

### Writeup / README
In this writeup, I decided to use pdf file. I also used this writeup as a readme file in grithub project. You're reading it!
### Data Set Summary & Exploration 
#### 1. I used the numpy library to calculate the dataset. Here is the summary of the traffic signs data set:
- The size of training set is 34799
- The size of the validation set is not provided so I will split it from the training set later.
- The size of test set is 12630
- The shape of a traffic sign image is (32,32,3)
- The number of unique classes/labels in the data set is 43
#### 2. Include an exploratory visualization of the dataset.
Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distribute for each class off traffic signs:

![Distribution of Images](https://github.com/loynin/Traffic_Sign_Classifier/blob/master/graph1.png)

### Design and Test a Model Architecture

#### 1. Model Selection:

Firstly, I used original dataset for training, validating, and testing. The result is not satisfied because the percentage of accuracy is around 80 percent. Then I try to create more fake data by creating more images per classes that has images less than 500. Using augment_img() function to change image contrast, size, color, and transform image from the original attribute to different attribute. The result still could not satisfy the requirement because the accuracy is below 90 percent. Finally, I decide to convert the images to grayscale and normalize the dataset; and I see the improvement on the result and I could get the accuracy above 93 percent.

Here is an example of a traffic sign images that have additionally created for classes that have less than 500 images.

![Distribution of Images](https://github.com/loynin/Traffic_Sign_Classifier/blob/master/augment_image.jpg)

### This is the new dataset
  
The new size of training set is 39239

Below first line of images are the converted grayscale images and the bottom line is the normalized image:

![Distribution of Images](https://github.com/loynin/Traffic_Sign_Classifier/blob/master/graph3.png)
 

#### 2. The model based on LeNet model with some tweak. 
My final model consisted of the following layers:

| Layer | Description |
|---|---|
| Input | 32x32x1 Grayscale image |
| Convolution 5x5 |	1x1 stride, VALID padding, outputs 28x28x6 |
| RELU |	|
| Max pooling |	2x2 stride, VALID padding, outputs 5x5x16 |
| Convolution 5x5 |	1x1 stride, VALID padding, outputs 1x1x400 |
| ReLu| |	
| Flatten layers |	Concatenate flattened layers |
| Dropout layer |	|
| Fully connected layer |	800 in, 42 out |

#### 3. This model is based on LeNet model. As it always, using the combination of the right parameters would be critical for model to perform well and improve accuracy. In this training, I have used the following parameters:
- 	Batch size = 100 
- 	Epochs = 25 
- 	mu = 0 
- 	sigma = 0.1
- 	rate = 0.001
- 	This model is using AdamOptimizer

#### 4. To get accuracy to be at lease 0.93, I have used LeNet model with some customizations. While using original LeNet, the accuracy is below 0.9. With some customization, the accuracy is above 0.93.

My final model results were:

- validation set accuracy of 0.988
- test set accuracy of 0.935

If a well known architecture was chosen:

- I used LeNet for the architecture of this training. On the other hand, I could not get the accuracy above 0.9 while using the original LeNet. I have to customize some of the function in order to improve the model accuracy.
- Follow by the instruction from the lesson in the classroom; I think that LeNet is a good model that can be used for the traffic sign classification.

### Test a Model on New Images
#### 1. Real Traffic Sign Selection:
Here are five German traffic signs that I found on the web:
         
The fourth image is difficult to classify because it doesn’t have in any class in the training dataset. Therefore, the training doesn’t know what class is the sign would be in.

#### 2. Accuracy of Model Prediction to real world traffic signs:

The model can predict four of the five images correctly. Therefore, the accuracy of the prediction is 80% compare to the training set of 93.4%. The fourth image is outside of the classes in the training set, therefore, it make sense that the model could not predict it while we doesn’t train it to recognize this image. 
Here are the results of the prediction:
Image	Prediction
Go straight or left	Go straight or left
Priority road	Priority road
Yield	Yield
Speed limit 40 km/h	End of no passing by vehicle over 3.5 metric tons
Speed limit 20 km/h	Speed limit 30 km/h

#### 3. For these real traffic sign images, the model predicts 60% correct. Only fourth and fifth image that are images that the model did not predicted accurately; Picture below shows how the result of the model’s prediction to real traffic signs. 

![Distribution of Images](https://github.com/loynin/Traffic_Sign_Classifier/blob/master/graph4.png)

For the first image, the model is relatively sure that this is a Go straight or left (probability of 1), and the image do contain a Go straight or left sign. The top five soft max probabilities were
Probability	Prediction
1	Go straight or left
1	Priority road
1	Yield
1 – Wrong prediction	End of no passing by vehicles over 3.5 metric tons
1 – Wrong prediction	Speed limit (30km/h)

Summary: 
In the field of self-driving car, traffic sign classification would be the first step to make sure that the car does follow the law. In this project, the prediction of traffic sign accuracy is about 93% but in the real world this result is not applicable to implement while the error is two large to take risk. This mean that there are sill many more traffic sign that the self-driving car could not recognize so the car could not make the decision in responding to the traffic signs and the situation. There are still many more to improve and to learn in this area in order to make the model work to get more accuracy to which we can implement in the real world situation. These improvements could be providing more training data, remodeling the model, augmenting the images and tweaking some more parameter what could improve accuracy of the prediction.
