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

## Performance Evaluation
The performance of each classifier will be evaluated with the top-1 accuracy metric, that is,  
$$ top-1 accuracy=\frac{number\ of\ correctly\ classified\ examples}{total\ number\ of\ test\ examples}$$
