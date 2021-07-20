# Image-classification-with-noise
## Introduction
The objective of this project is to build an transition matrix estimator and two classification algorithms that are robust to label noise.  

Three input datasets are given. For each dataset, the training and validation data contains class-conditional random label noise, whereas the test data is clean. I build at least two different classifiers trained and validated on the noisy data, that have a good classification accuracy on the clean test data. I build an transition matrix estimator to estimate the transition matrix. Then, employ estimated transition matrix for classification. 

## Dataset Description
#FashionMINIST0.5.npz
* Number of the training and validation examples n = 18000.
* Number of the test examples m = 3000.
* The shape of each example image shape = (28 × 28).

#FashionMINIST0.6.npz
* Number of the training and validation examples n = 18000. 
* Number of the test examples m = 3000.
* The shape of each example image shape = (28 × 28).

#CIFAR.npz
* Number of the training and validation examples n = 15000. 
* Number of the test examples m = 3000.
* The shape of each example image shape = (32 × 32 × 3).

## Load dataset
```
dataset = np.load($FILEPATH) Xtr val = dataset[’Xtr’]
Str val = dataset[’Str’]
Xts = dataset[’Xts’]
Yts = dataset[’Yts’]
```

>`Traning data`: The variable `Xtr_val` contains the features of the training and validation data. The shape is (n, image shape) where n represents the total number of the in- stances. The variable `Str_val` contains the noisy labels of the n instances. The shape is (n, ). For all datasets, the class set of the noisy labels is {0, 1, 2}.`Note that` I am independently and randomly sample 80% of the n examples to train a model and use the rest 20% examples to validate the model.

>`Test data`: The variable `Xts` contains `features` of the test data. The shape is (m, image shape), where m represents the total number of the test instances. The variable `Yts` contains the `clean labels` of the m instances. The class set of the clean labels is also {0, 1, 2}.

## Method

### Performance Evaluation
The performance of each classifier will be evaluated with the top-1 accuracy metric, that is,  
$$ top-1 accuracy=\frac{number\ of\ correctly\ classified\ examples}{total\ number\ of\ test\ examples}$$


### Transition matrix
Noisy transition matrix (T) is a matrix of the probabilities that clean labels (Y) flip into noisy labels (Y^). This is important for learning correct classification. In this report, we focus on class-dependent transition matrix, which can be described as:  
![image](https://github.com/germaine-wu/Image-classification-with-noise/blob/main/image/Transition%20matrix.png)
</br>

For the first two datasets, they exists transition matrix T . It shows below:
For FashionMINIST0.5 dataset:
![image](https://github.com/germaine-wu/Image-classification-with-noise/blob/main/image/The%20transition%20matrix%20of%20FashionMINIST0.5%20dataset.png)

For FashionMINIST0.6 dataset:
![image](https://github.com/germaine-wu/Image-classification-with-noise/blob/main/image/The%20transition%20matrix%20of%20FashionMINIST0.6%20dataset.png)

For the third dataset, the transition matrix T is unknown. We find max probabilities for each label. For example, if X = x2 (sample x2) has the highest probability in label Y = 1, P(X = x2) will become the first column of transition matrix. Then, we use transition matrix in test function. We calculate the probabilities that each sample belongs to each label in test processing. We use noisy probabilities to find anchor points and estimate a transition matrix. By using the formula we introduced in section 3.2, we estimate clean probabilities, and compare the clean probabilities to target label.

By using this method, for dataset CIFAR, the transition matrix of DaiNet7 CNN model is:
![image](https://github.com/germaine-wu/Image-classification-with-noise/blob/main/image/The%20transition%20matrix%20of%20DaiNet7%20CNN%20model.png)

The transition matrix of CNN2 model is:
![image](https://github.com/germaine-wu/Image-classification-with-noise/blob/main/image/The%20transition%20matrix%20of%20CNN2%20model.png)

###  Backward Learning
In this report, we only use backward learning rather than forward learning. Backward learning is one of methods to utilize transition matrix to improve classifier’s performance on noise dataset. The algorithm firstly trained a neural network model on noise data set, and then using this neural network model to verify test set.

###  Classifier Architecture
We applied different convolutional neural networks (CNN) based architecture. It can effectively reduce the dimension of big images into small data, and extract and retain image features. A CNN consists of an input and an output layer, as well as multiple hidden layers. The hidden layers consists of one or more convolutional layers, activation functions (like ReLU layer), and pooling layers. Finally, it goes fully connected layers. Convolutional layers are responsible for extracting local features in images. The input images become abstracted to a feature map by passing through a convolutional layer. It usually connects with a ReLU layer. The pooling layer is used to significantly reduce parameter magnitude (dimension reduction) to avoid overfitting. The full connected layer is the part of a traditional neural network that outputs the desired result.

The first FashionCNN1 model has three block. The first two are convolutional and the last one is fully connected layer. The detail of each layer is shown:
![image](https://github.com/germaine-wu/Image-classification-with-noise/blob/main/image/FashionCNN1%20model.png)

The first FashionCNN2 model has five block. The first two are convolutional and the last three is fully connected layer. The detail of each layer is shown:
![image](https://github.com/germaine-wu/Image-classification-with-noise/blob/main/image/FashionCNN2%20model.png)

The first DaiNet7 CNN1 has seven layers with weights. The first six are convolutional and the last one is fully connected layer. The detail of each layer is shown:
![image](https://github.com/germaine-wu/Image-classification-with-noise/blob/main/image/DaiNet7%20CNN1%20base%20architecture.png)

The second CNN2 has six layers with weights. The first three are convolutional and the remaining three is fully connected layer. The detail of each layer is shown:
![image](https://github.com/germaine-wu/Image-classification-with-noise/blob/main/image/CNN2%20base%20architecture.png)

## Experiment
### Dataset
FashionMINIST 0.5 and 0.6 both contain 18000 training examples and 3000 test examples. Each sample is in the shape (28, 28), showed as below Figure 1(a) and 1(b). The CIFAR training data set includes 15000 samples and the test data set includes 3000 samples, and the shape of each sample is in the shape (32, 32, 3), showed as below:
![image](https://github.com/germaine-wu/Image-classification-with-noise/blob/main/image/The%20picture%20of%20datasets.png)


Firstly, data are encapsulated as tensor types by using superclass Dataset, and randomly choose 80% of training set as training set and the remaining 20% was used as the validation set. Then, function DataLoader is used to set batch size. We set batch size of all datasets to 100.

### Setup

We use two different CNN models (mentioned in section 3.3) to train both FashionMINIST0.5 and FashionMINIST0.6. Using other two different CNN models (mentioned in section 3.3) to train dataset CIFAR. For the FashionMINIST0.5, the epoch in first model is set as 50 and that in the second method is set as 10. The epoch in first and second model of FashionMINIST 0.6 is 10 respectively. The epoch settings of the first and second methods of CIFAR are 25 and 10 respectively. The detail is showed below:
![image](https://github.com/germaine-wu/Image-classification-with-noise/blob/main/image/Experiment%20setting.png)

### Results

From Table 6, we can see both both datasets FashionMINIST0.5 and FashionMINIST0.6 have high accuracy in model FashionCNN1. The accuracy without multiplying transition matrix achieve to 93.3% and 90% respectively in datasets FashionMINIST0.5 and FashionMINIST0.6. This means that model FashionCNN1 is ready robust to the two datasets. Therefore, when we add transition matrix, there is no change or slight change in accuracy. We can see, the accuracy of FashionMINIST0.5 increases by 1%, while that of FashionMINIST0.6 has not changed. This means that transition matrix is slightly robust to dataset FashionMINIST0.6, but it does not have obviously robust to FashionMINIST0.6.

The test results of CIFAR are not ideal. For the first model (DaiNet7), there is a 1% increase if we multiply transition matrix T3. This means that transition matrix has some effects to noisy label. However, if we use the transition matrices T1 and T2, the accuracy is higher than T3, which means the transition matrix that we estimate is erroneous. For the second model (CNN2), there is a large increase if we multiply transition matrix. The accuracy boosts from 60.83% to 67.4%, which shows that transition matrix has a large impact to label noise. If we use the transition matrices T1 and T2, the accuracy remain unchanged. Therefore, the transition matrix that we estimate in the second model is good.
![image](https://github.com/germaine-wu/Image-classification-with-noise/blob/main/image/%20The%20mean%20and%20the%20standard%20derivation%20of%20the%20test%20accuracy.png)

## Conclusion
This report use four CNN models and transition matrices to deal with class-conditional random label noise. The models FashionCNN1 and FashionCNN2 are robust to datasets FashionMINIST 0.5 and FashionMINIST 0.6. They can clearly distinguish between true label and wrong label, so the accuracy are high. The models DaiNet7CNN1 and CNN2 cannot accurately identify the noisy label, but the transition matrices can improve accuracy of the model. The future may related to how to find a module that has suitable structure for the label-noise training set and its transition matrix.


