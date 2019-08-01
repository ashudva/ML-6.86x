# ML-6.86x
Machine Learning Projects Repository

# Projects
1. Review Analyzer
2. Digit Recognition (Non-Linear Classifier & Kernel)
3. Digit Recognition (Neural Networks & Deep Learning)
4. Collaborative Filtering
5. Text Based Game

## Review Analyzer

The goal of this project is to design a classifier to use for **sentiment analysis of product reviews**. Our training set consists of reviews written by *Amazon* customers for various food products. The reviews, originally given on a 5 point scale, have been adjusted to a +1 or -1 scale, representing a positive or negative review, respectively.

Review | 	label
-------- | --------
Nasty No flavor. The candy is just red, No flavor. Just plan and chewy. I would never buy them again | -1
YUMMY! You would never guess that they're sugar-free and it's so great that you can eat them pretty much guilt free! i was so impressed that i've ordered some for myself (w dark chocolate) to take to the office. These are just EXCELLENT! | +1

### Use classifiers on the food review dataset, using some simple text features.

>#### In order to automatically analyze reviews we will implement & compare the performance of the algorithms :
>#### 1. Perceptron Algorithm
>*Training Accuracy = 0.8157 , Validation Accuracy = 0.7160 , Best T = 25*

>#### 2. Average Perceptron Algorithm
>*Training Accuracy = 0.9728 , Validation Accuracy = 0.7980 , Best T = 25*

>#### 3. Pegasos Algorithm
>*Training Accuracy = 0.9143 , Validation Accuracy = 0.7900 , Best T = 25, Best l = 0.01*

#### Most Explanatory Words for positively labeled reviews:
1. Delecious
2. Great
3. !
4. Best
5. Perfect
6. Loves
7. Wonderful
8. Glad
9. Love
10. Quickly

# Predictions
>making predictions usig **Pegasos** `T = 25 & L = 0.01`
#### 1. Normal features pegasos
Training Accuracy = 0.9185
Test Accuracy = 0.8020
#### 1. Stopword features pegasos
Training Accuracy = 0.9157
Test Accuracy = 0.8080
#### 1. Stopword w/o binarize features pegasos
Training Accuracy = 0.8928
Test Accuracy = 0.7700


## Digit Recognition (Non-Linear Classifier & Kernel)

**Digit Recognition** using the *MNIST (Mixed National Institute of Standards and Technology)* database.

[MNIST Dataset Wiki](https://en.wikipedia.org/wiki/MNIST_database)

The MNIST database contains binary images of handwritten digits commonly used to train image processing systems. The digits were collected from among Census Bureau employees and high school students. The database contains 60,000 training digits and 10,000 testing digits, all of which have been size-normalized and centered in a fixed-size image of 28 × 28 pixels. Many methods have been tested with this data-set and in this project, classify these images into the correct digit.

Sample Digit Images:

![Image of Digit 6](https://prod-edxapp.edx-cdn.org/assets/courseware/v1/03f49ce9ab37fa92d84b0c9e70542014/asset-v1:MITx+6.86x+1T2019+type@asset+block/images_6.png) ![Image of Digit 8](https://prod-edxapp.edx-cdn.org/assets/courseware/v1/e7123412da031f62e082afb10bdfa655/asset-v1:MITx+6.86x+1T2019+type@asset+block/images_8.png)  ![Image of Digit 8](https://prod-edxapp.edx-cdn.org/assets/courseware/v1/280748cc6f7447b43db835bf0c1700d8/asset-v1:MITx+6.86x+1T2019+type@asset+block/images_x.png) ![Image of Digit 6](https://prod-edxapp.edx-cdn.org/assets/courseware/v1/b56e40dfe8c00d6c9b54956f21e04f92/asset-v1:MITx+6.86x+1T2019+type@asset+block/images_6-2.png)

> ## Learning Method 1 - *Linear Regression : Closed form solution*
>
>We can apply a linear regression model, as the labels are numbers from 0-9 .
>
>Function `closed_form` that computes this closed form solution given the features  `X` , labels  `Y`  and the regularization parameter  `λ` .
>
>Calculate test error of linear regression algorithm for different  λ using function `compute_test_error_linear(test_x, Y, theta)` .
>
>Results: No matter what  λ  factor we try, the test error is large when we use Linear regression


## Digit Recognition (Neural Networks & Deep Learning)

>This project uses PyTorch for *implementing neural networks* & SciPy to handle *Sparse Matrices*
