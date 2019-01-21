# Forensic-Project
To find similarity between the handwritten samples of the known and the questioned writer in forensics

To solve criminal cases dealing with evidence provided by handwritten documents such as wills and ransom notes. The project requires us to apply machine learning to solve the handwriting comparison task in forensics. 

Each instance in the CEDAR "AND" training data consists of set of input features for each hand-written "AND" sample. The features are obtained from two different sources:
1. Human Observed features: Features entered by human document examiners manually
2. GSC features: Features extracted using Gradient Structural Concavity (GSC) algorithm.

The target values are scalars that can take two values 1:same writer, 0:different writers.

Source of Dataset:

The dataset uses "AND" images samples extracted from CEDAR Letter dataset. The CEDAR dataset consists of handwritten letter manuscripts written by writers. Each of the writer has copied a source document thrice. Image snippets of the word "AND" were extracted from each of the manuscript using transcript-mapping function of CEDAR-FOX.

Types of Datasets:
Based on feature extraction process, we have been provided two datasets:
1.) Human Observed Dataset

The Human Observed dataset shows only the cursive samples in the data set, where for each image the features are entered by the human document examiner.There are total of 18 features for a pair of handwritten "AND" sample (9 features for each sample). The dataset is named as "HumanObservedDataset.csv". The entire dataset consists of 2 image ids, 18 features and a target value for each handwritten sample pairs (rows).

2.) GSC Dataset using Feature Engineering

Gradient Structural Concavity algorithm generates 512 sized feature vector for an input handwritten "AND"image. GSC algorithm extracts 192 binary gradient features, 192 binary structural features and 128 concavity features. There are total of 1024 features for a pair of handwritten "AND" sample (512 features for each sample). The dataset is named as "GSCDataset.csv".The entire dataset consists of 2 image ids, 1024 features and a target value for each handwritten sample pairs (rows).

Under two settings, we have to apply the machine learning algorithms:
1: Feature Concatenation (18 Features)
2: Feature subtraction (9 Features)

The above means, I have to create four datasets by feature concatenation and feature subtraction and for both datasets, I will apply the machine learning algorithms.
So, 4 different input datasets are:
(1) Human Observed Dataset with feature concatentation
(2) Human Observed Dataset with feature subtraction
(3) GSC Dataset with feature concatentation
(4) GSC Dataset with feature subtraction

For HumanObserved dataset:
There are three files "HumanObserved-Features-Data.csv", "same_pairs.csv" and "diffn_pairs.csv".
For GSC Dataset:
There are "GSC-Features.csv", "same_pairs_gsc.csv" and "diffn_pairs_gsc.csv".

To make four datasets I have to extract features values and Image Ids from the data(given CSV files for both Human Observed and GSC).

Here I have used three models 1) Linear Regression 2) Logistic regression 3) ANN (Artificial Neural Network).

Conclusion
During making this project, I came to know important concept an algorithm like linear regression,logistic regression and ANN. The linear regression does not give good prediction on discrete outputs. So, whenever we see labels are discretized, then it will be wiser
step to choose other classification model and ANN than Linear regression model.
