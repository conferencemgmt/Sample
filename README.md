---
date: 2020-11-15 15:00:00
layout: post
title: Introduction To Scikit-Learn
subtitle: "Welcome to our 1st Blog in Scikit-learn Series"
description: >-
  `Scikit-learn` is an open source machine learning library that supports supervised and unsupervised learning. It also provides various tools for model fitting, data preprocessing, model selection and evaluation, and many other utilities.
image: >-
  https://lh3.googleusercontent.com/9G0cbQzN7wsdAX0NOGc9xyNWQRMKUlRG_slO4R6lzy4ejPKf0IlalRTTVw70Bn3gYJzy0ahPzKcmWnxx3L3z45LhQzyjmaBNh22vVqEt0Z9SHb1onwfUwdcy7AIKnL1EgLKt3uiVD1ur341WyyqErZ3hXGW_qSfZNVP5Ubxs0wny9chpwGlUab6Q6MJcPXMuI9W6oMtWI0mrD1Q_QeNL2qIyQJ4QroO8O8-9JqNq-jEmWHp7kBd_yPzaxIPtWodq6BbngrVvOUAft7nZoVcDfN0tVCxIshEvefCJoklxX6amg5Gm6qFq2-Roq3aLAEsosKgZisUclGMQaUtnSRNMwO_VKTvs0teayq5axNi77Bkje3W-GBK_vMkia-uCdVdR59Py91j-Lej6nBbi4BUbhJO_KQ4Jp1m-71CGh2iOr3A4GXha7U7J43IatBa-9N9cXEb3T1XnCGIM40ICRVpGDfsoPesvI7UO-_uuZjWHi2rStvkxGDfzULwJwvK9MMi9iXZtcAl8d0MqRgeyhgHAXToSZ3m3nvYm60ux1VdmZgF3d7tjxbhJYDyV0PIn9vn29NihWwTbz4DHaXGLII6wH7nqONXGeiPfVsZVsCLw44_GiSxSM9tYPbuT9zSVLOTitPEQx4slymd_1xePyLKA7U2fTLiuLW-9U6ZTj2WguzB14zOYPCmCetUpv9yPVQ=w1000-h350-no?authuser=0
optimized_image: >-
  https://lh3.googleusercontent.com/9G0cbQzN7wsdAX0NOGc9xyNWQRMKUlRG_slO4R6lzy4ejPKf0IlalRTTVw70Bn3gYJzy0ahPzKcmWnxx3L3z45LhQzyjmaBNh22vVqEt0Z9SHb1onwfUwdcy7AIKnL1EgLKt3uiVD1ur341WyyqErZ3hXGW_qSfZNVP5Ubxs0wny9chpwGlUab6Q6MJcPXMuI9W6oMtWI0mrD1Q_QeNL2qIyQJ4QroO8O8-9JqNq-jEmWHp7kBd_yPzaxIPtWodq6BbngrVvOUAft7nZoVcDfN0tVCxIshEvefCJoklxX6amg5Gm6qFq2-Roq3aLAEsosKgZisUclGMQaUtnSRNMwO_VKTvs0teayq5axNi77Bkje3W-GBK_vMkia-uCdVdR59Py91j-Lej6nBbi4BUbhJO_KQ4Jp1m-71CGh2iOr3A4GXha7U7J43IatBa-9N9cXEb3T1XnCGIM40ICRVpGDfsoPesvI7UO-_uuZjWHi2rStvkxGDfzULwJwvK9MMi9iXZtcAl8d0MqRgeyhgHAXToSZ3m3nvYm60ux1VdmZgF3d7tjxbhJYDyV0PIn9vn29NihWwTbz4DHaXGLII6wH7nqONXGeiPfVsZVsCLw44_GiSxSM9tYPbuT9zSVLOTitPEQx4slymd_1xePyLKA7U2fTLiuLW-9U6ZTj2WguzB14zOYPCmCetUpv9yPVQ=w1000-h350-no?authuser=0
category: blog
tags:
  - welcome
  - python
  - blog
  - scikit-learn
author: Ketan Bansal
---

# Introduction

Scikit learn is a machine learning library for python programming language which offers various important features for machine learning such as classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to inter-operate with the python numerical and scientific libraries like **Numpy** and **SciPy**.

_We will discuss each algorithm and its implementation with codes in detail later in the second part of this series._

### Supervised Algorithms In Scikit-Learn

Since you are already familiar with machine learning you already know that there are two types of algorithms i.e supervised and unsupervised algorithms.
So, we will see what scikit-learn has to offer in supervised algorithms.

_The problem of supervised learning can be broken into two_ :

![alt text](https://lh3.googleusercontent.com/DxdPPzBM69RJvSGtKn7HulpYX-2IhDctSprGrOSxEaLe2oreh1GFhJ90rp4cLVR0gWMN-G4e9BMCAMWJi1knHwkk_n_OTjSxMPIFbIgM1hLQUYM92VrK94RWIVyqNZn1fvVjnYKGbuZvF7qfOw6n9kpvquoSFKvZ5ZxA3Gh6urxmsoFhXftQT1ddWl9SQbjGcPTbODxebc7v2d-AN2tajnBdtWRJZCfvpBk4HUwMqHJW-mCcLHGUItRDqEpIIkl4yW2EJ4wppECTVzKSyExOMdBhEiVdA9AlT_rThggzgd3PH49-2cdwhbpf8KDTegPQ4qEV5VGSfwg6uCJ9nMJ6WFGfVzYEuRrCEg39f5uwXsL7JPFRFEWwv371duoddm7gHNqdzLnGmZ5GRYCa_kkug6sxB6Ln-I1pQQG1wsaTSsiOdCpNX5G2SSDNqdqR5YpMUdgekhMpIczsVAW7azxK5BQ57yZxB2fgozsLJ2aAylTLoocs8VVo4IgMkMZP0eR34RPphw3BRqhe_epvOtZffh61GY0rIvIIoKjYQqbd2L8LONQZ4wvrhGsqQ-7djuZVI76214FqeX-icdyvV2qJRVHfCly3oEhSUwpi8ONaRES_h7owLyeGbdpiBq13AGiQYZmegDnRc3b17o6jqyZVZXuYx81JatVnZ-m1Y0P6vYgKM3sdp8QrqdXtL2KXPg=w608-h308-no?authuser=0)

**Classification**: Samples belong to two or more classes, and we want to learn from already labeled data on how to predict the class of unlabeled data. An example would be the handwritten digit recognition example, in which the aim is to assign each input vector to one of a finite number of discrete categories. Another way to think of classification is as a discrete (as opposed to continuous) form of supervised learning where one has a limited number of categories and for each of the n samples provided, one is to try to label them with the correct category or class.

Checkout the implementation of the above example: 
_https://colab.research.google.com/drive/1LKv9OBo59e0tf43ZHuZy2br-5cXTzAoX?usp=sharing_

**Regression**: If the desired output consists of one or more continuous variables, then the task is called regression. An example of a regression problem would be the prediction of the diabetes in which input consist of the age, sex, body mass index, average blood pressure, etc.

Checkout the implementation of the above example: 
https://colab.research.google.com/drive/1Qav2lstI1NasHrRhhvpwco_EXzy8f3_P?usp=sharing

_**It's okay if you don't understand the code, we will be discussing it in detail later.**_

#### Scikit Learn supports following models :
- Generalized Linear Models
- Kernel ridge regression
- Support Vector Machines
- Stochastic Gradient Descent
- Linear and Quadratic Discriminant Analysis
- Naive Bayes
- Decision Trees
- Ensemble methods
- Multiclass and multilabel algorithms
- Feature selection
- Nearest Neighbors
- Gaussian Processes
- Cross decomposition
- Semi-Supervised
- Isotonic regression
- Probability calibration
- Neural network models (supervised)

### Unsupervised Algorithms In Scikit-Learn

Now, let us see what scikit learn offers us in unsupervised algorithms.

![alt text](https://lh3.googleusercontent.com/Wy4g2pdRGr4lR2dq65TgSoyurKOFdfzuiStCg6M9t-4dikRdP9s1owllUYRqfyq3Nu3ZYoYnKK6-7VdSzWnZgTmmdANGYqteQU-6UeI16Q41Zx3FEKnVXb90c5V1INJcBA-iSSAWmrqmoHJXgBz-3V4cK-ruCf7p4VX5o1TdPSNWsOCxjHgn6V4W6oXAf6XrI0KIC0CMUkcuJz4wyTlzR_Z3hjYNz-VgAjLp_SES4FM9YAQhJeUJlUVZGBU9xagAxNkSx2jgkjBCRA_8t4T5OetS9xu2wodMJ9Ef1E2Rx1NtUOjc8i7TW8BTfmtU5n--x8ukce2WKpTifDESjYfX-NLreiH5N1sGKxrS2CW9D3_c7Bzp2B0BQJagOCYJGAFFbF_uZzAl4PEKvNuD2a6WdYpno5Cqrh87xe7GQgirzC9hTrc0P4v3VwV6wF8oUlwz7P6dvKWg32twvTjRU2LM7DFqYwhXGbtPMV5FQCwE3AFhGjYMPC6a-53wqif-0uWK6AXKCVU2IsMdp9VPdbI43IusEetfHaZrGMN_Ybx9OWqPB9eS7EWr2_JRIKzSI_FwSqVw2rzxLdfs0D4yn-PsIRxuv_uX9DebTafsl3Ec3pGgxMuuda-jlka52l4eWbcOLdZILOxXHjaZO2-KPRnsPNT771v211-qSB8LS9B8LD4-SM1zSFqCEb4eUpx2GQ=w561-h283-no?authuser=0)

In this the training data consists of a set of input vectors without any corresponding target values/labels. The goal in such problems is to discover groups of similar data within the data, where it is called **clustering**.

Checkout the implementation of K-means clustering on Handwritten digits example: 
https://colab.research.google.com/drive/1Tj3TDpFBq2iYIIcgMTRzsiLJec6QJjhx?usp=sharing

_**It's okay if you don't understand the code, we will be discussing it in detail later.**_

##### Scikit Learn supports these models :

- Gaussian mixture models
- Decomposing signals in components (matrix factorization problems)
- Covariance estimation
- Novelty and Outlier Detection
- Manifold learning
- Clustering
- Biclustering
- Density Estimation
- Neural network models (unsupervised).



### Model Selection and Evaluation

As we know learning the parameters of a prediction function and testing it on the same data is a methodological mistake or it can be called as cheating : a model that would just repeat the labels of the samples that it has just seen would have a perfect score but would fail to predict anything useful on yet-unseen data. This situation is called **overfitting**. To avoid it, it is common practice when performing a (supervised) machine learning experiment to hold out part of the available data as a test set X_test, y_test.

![alt text](https://lh3.googleusercontent.com/tcA5I62ye_bXJY_bXpWq2cZBxayQRwwXCJNM6Ezb-d4tEj-rczfLbnvI4pnMv4ZPpnBdQCr-QW048Am19f4FndrSkt-qPno02HhmLN5fRj0I6WHcdWwmzz8lmvae98QC65uP3YOXB6Zi65A9C_KU_YS0MQkoKDqecU1HEIDGB3gEBmxXQ96Cg54jifbRbGPaRy40selcZ8WTnM9wNy2tGAvqj3fi1mm7MnrImyIJVyY2wscl_Cp7xDPFG-YrNdKuaH1nVnBUQT8etXYputJvSCGrxvDX6xXxqbPdZ22YdWown8ZoOHk7KB_JRe0bjKHRcqi8Ec6Cw8zkIC-gAZkGS7O18Z0eaxv24h8xwS1TWG8cx30QZSqtO_hLd-HogYSNSQTU2mxSo4tCf_i7eFoeWVrNnZZFjie4quFbZ9WMvCp5j_KY9HbRgSrC18eveo-HK_gew2RSE8qGrijfgCU0ZA-cHDntbu6QUDCSU33e3lvk9g-vwl7VkuWtfVFn7MJjx4RE39wJ_jB1Xore8cXiCc__rSGfIHKvHk1EQWxGt_cN8R6Uq8hLjNjazcMcyz_bhoupU9Q6YpWmAvfdpGeUWWiJ2qRG4R1ByXkrHJeddhd-LAado9ey4eCYhoDwj1O6tH26MdcNW9vDHxC5xjcSrVd7KW-aojxRMvOF8NYjnfoR4ueJmg-prU8-TqGkKA=w1400-h500-no?authuser=0)

_The above image shows three cases underfitting, ideal and overfitting scenarios respectively._

#### Model selection contains the following :
- Cross-validation: evaluating estimator performance
- Tuning the hyper-parameters of an estimator
- Model evaluation: quantifying the quality of predictions
- Model persistence
- Validation curves: plotting scores to evaluate models

### Dataset transformations

These are represented by classes with a fit method, which learns model parameters (e.g. mean and standard deviation for normalization) from a training set, and a transform method which applies this transformation model to unseen data. fit_transform may be a more convenient and efficient for modelling and transforming the training data simultaneously.

It has following sub-categories :
- Pipeline and FeatureUnion: combining estimators
- Feature extraction
- Preprocessing data
- Unsupervised dimensionality reduction
- Random Projection
- Kernel Approximation
- Pairwise metrics, Affinities and Kernels
- Transforming the prediction target (y)

### Dataset Loading Utilities

The `sklearn.datasets` package embeds some small yet useful datasets.

- The Olivetti faces dataset
- The 20 newsgroups text dataset
- Downloading datasets from the mldata.org repository
- The Labeled Faces in the Wild face recognition dataset
- Forest covertypes
- RCV1 dataset
- Boston House Prices dataset
- Breast Cancer Wisconsin (Diagnostic) Database
- Diabetes dataset
- Optical Recognition of Handwritten Digits Data Set
- Iris Plants Database
_and many more.._

### Resources

- [scikit-learn documentation](https://scikit-learn.org/)
- [scikit-learn Repository](https://github.com/scikit-learn/scikit-learn)

### Continued in next Week....
This was a brief intro of what we will be discussing in next few weeks, stay tunned..
