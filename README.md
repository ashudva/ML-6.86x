# ML-6.86x [![BCH compliance](https://bettercodehub.com/edge/badge/ashudva/ML-6.86x?branch=master)](https://bettercodehub.com/) [![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/ashudva/ML-6.86x)
Machine Learning Projects Repository

# Projects
1. Review Analyzer
2. Digit Recognition (Non-Linear Classifier & Kernel)
3. Digit Recognition (Neural Networks & Deep Learning)
4. Collaborative Filtering via Gaussian Mixtures
5. Home World Game (Autonomous Game Agent using RL)

## Review Analyzer

The goal of this project is to design a classifier to use for **sentiment analysis of product reviews**. Our training set consists of reviews written by *Amazon* customers for various food products. The reviews, originally given on a 5 point scale, have been adjusted to a +1 or -1 scale, representing a positive or negative review, respectively.

Review | 	label
-------- | --------
Nasty No flavor. The candy is just red, No flavor. Just plan and chewy. I would never buy them again | -1
YUMMY! You would never guess that they're sugar-free and it's so great that you can eat them pretty much guilt free! i was so impressed that i've ordered some for myself (w dark chocolate) to take to the office. These are just EXCELLENT! | +1

### Learning Algorithms
Project involves comparative performance study of three learning algorithms w.r.t sentiment analysis.
1. Perceptron Algorithm 
2. Average Perceptron Algorithm
3. Pegasos Algorithm

### Results
Pegasos Algorithm performs best of all threee algorithms. for detailed results head to [Review-Analyzer](/Review-Analyzer/Readme.md).
## Digit Recognition (Non-Linear Classifier & Kernel)

**Digit Recognition** using the *MNIST (Mixed National Institute of Standards and Technology)* database.

[MNIST Dataset Wiki](https://en.wikipedia.org/wiki/MNIST_database)

The MNIST database contains binary images of handwritten digits commonly used to train image processing systems. The digits were collected from among Census Bureau employees and high school students. The database contains 60,000 training digits and 10,000 testing digits, all of which have been size-normalized and centered in a fixed-size image of 28 × 28 pixels. Many methods have been tested with this data-set and in this project, classify these images into the correct digit.

Sample Digit Images:

![Image of Digit 6](https://prod-edxapp.edx-cdn.org/assets/courseware/v1/03f49ce9ab37fa92d84b0c9e70542014/asset-v1:MITx+6.86x+1T2019+type@asset+block/images_6.png) ![Image of Digit 8](https://prod-edxapp.edx-cdn.org/assets/courseware/v1/e7123412da031f62e082afb10bdfa655/asset-v1:MITx+6.86x+1T2019+type@asset+block/images_8.png)  ![Image of Digit 8](https://prod-edxapp.edx-cdn.org/assets/courseware/v1/280748cc6f7447b43db835bf0c1700d8/asset-v1:MITx+6.86x+1T2019+type@asset+block/images_x.png) ![Image of Digit 6](https://prod-edxapp.edx-cdn.org/assets/courseware/v1/b56e40dfe8c00d6c9b54956f21e04f92/asset-v1:MITx+6.86x+1T2019+type@asset+block/images_6-2.png)


### Learning Algorithms
##### Linear Regression (Closed form solution)

1. We can apply a linear regression model, as the *labels are numbers from 0-9* .
2. Function `closed_form` that computes this closed form solution given the features  `X` , labels  `Y`  and the regularization parameter  `λ` .
3. Calculate test error of linear regression algorithm for different  `λ` using function `compute_test_error_linear(test_x, Y, theta)` .
4. Results: No matter what `λ`  factor we try, the test error is large when we use Linear regression

## Digit Recognition (Neural Networks & Deep Learning)

>This project uses PyTorch for *implementing neural networks* & SciPy to handle *Sparse Matrices*


## Collaborative Filtering via Gaussian Mixtures

Our task is to build a mixture model for collaborative filtering. Given a data matrix containing movie ratings made by users where the matrix is extracted from a much larger Netflix database. Any particular user has rated only a small fraction of the movies so the data matrix is only partially filled. The goal is to predict all the remaining entries of the matrix.

We will use mixtures of Gaussians to solve this problem. The model assumes that each user's rating profile is a sample from a mixture model. In other words, we have  K  possible types of users and, in the context of each user, we must sample a user type and then the rating profile from the Gaussian distribution associated with the type. We will use the Expectation Maximization (EM) algorithm to estimate such a mixture from a partially observed rating matrix. The EM algorithm proceeds by iteratively assigning (softly) users to types (E-step) and subsequently re-estimating the Gaussians associated with each type (M-step). Once we have the mixture, we can use it to predict values for all the missing entries in the data matrix.
