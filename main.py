import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm_notebook
from keras.utils import np_utils
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
% matplotlib inline

# Making feature dataset after concatenation

# Importing HumanObserved-Features-Data.csv file
humanObservedFeaturesData = pd.read_csv("HumanObserved-Features-Data.csv")
humanObservedFeaturesData.head()
humanObservedFeaturesData.shape

humanObservedFeaturesData = humanObservedFeaturesData.drop('Unnamed: 0',axis=1)
humanObservedFeaturesData.head()
humanObservedFeaturesData.shape

# Importing same_pairs.csv file

samePair = pd.read_csv("same_pairs.csv")
samePair.head()
samePair.shape

print(samePair['img_id_A'].unique().shape)  # It tells the number of distinct images in img_id_A column
print(samePair['img_id_B'].unique().shape)  # It tells the number of distinct images in img_id_B column

# Both columns have 564 distinct images out of 791 images
# Importing diffn_pairs.csv file

diffnPair = pd.read_csv("diffn_pairs.csv")
diffnPair.head()
diffnPair.shape

# To make my feature dataset, here I am taking random 791 samples from diffnPair dataframe so that I can concatenate same pair
# and different pair along row axis.

humanObsFeatures = pd.concat([samePair, diffnPair[:791]])
humanObsFeatures.head()  # We can see in the target column, series of 1's is appearing and then series of 0's
humanObsFeatures.shape  

hod = humanObsFeatures.values
np.random.shuffle(hod)          # Here I have shuffled the samples along row axis
print(hod)     
# We can clearly infer from above output that hod is shuffled.

columnIndex = humanObsFeatures.columns
columnIndex
humanObsFeatures =  pd.DataFrame(data = hod,columns=columnIndex)
humanObsFeatures.head()

# To make human observed dataset after concatenation of features of image_A and image_B, we have to fetch features of image A
# and B from "humanObservedFeaturesData" dataframe.
img_id_A = humanObsFeatures['img_id_A'].values
img_id_B = humanObsFeatures['img_id_B'].values
humanObservedFeaturesData.head()

# To fetch the features we have to reset the index

features = humanObservedFeaturesData.set_index(keys='img_id')
features.head()

# Fetching Images_A features

img_id_A_features = features.loc[img_id_A]
img_id_A_features.head()
img_id_A_features.shape

# Fetching Images_B features

img_id_B_features = features.loc[img_id_B]
img_id_B_features.head()
img_id_B_features.shape

imageIds = humanObsFeatures[['img_id_A', 'img_id_B']].values
target = humanObsFeatures[['target']].values

subtractedFeatures = np.abs(img_id_A_features.values - img_id_B_features.values)
subtractedFeatures
subtractedFeatures = np.hstack((imageIds, subtractedFeatures, target)) 
subtractedFeatures
subtractedFeatures.shape
Indexcol = ['img_id_A', 'img_id_B'] + list(img_id_A_features.columns) + ['target']
Indexcol
subtractedFeatures = pd.DataFrame(subtractedFeatures, columns = Indexcol)
subtractedFeatures.head()
subtractedFeatures.shape
columnIndex = ""

for i in range(1,19):
    columnIndex = columnIndex + 'f' + str(i) + ' '
columnIndex
img_id_A_features.values
img_id_B_features.values
ConcatenatedFeatures = np.hstack((imageIds, img_id_A_features.values, img_id_B_features.values, target)) 
ConcatenatedFeatures
col = ['img_id_A', 'img_id_B']+columnIndex.split() + ['target']  # Index of Columns 
print(col)
ConcatenatedFeatures = pd.DataFrame(ConcatenatedFeatures, columns = col)
ConcatenatedFeatures.head()
ConcatenatedFeatures.shape

# Now we have both datasets ConcatenatedFeatures and subtractedFeatures
subtractedFeatures.shape

# Generation of target vector from ConcatenatedFeatures and subtractedFeatures for human observed data and GSC

def getTargetVector(file):
    targetVec = file['target'].values
    return targetVec


# Generation of raw data matrix from ConcatenatedFeatures and subtractedFeatures for human observed data and GSC

def rawDataMatrix(file):
    rawData = file[file.columns].values
    return rawData[:,2:len(rawData[0])-1].T
    
# Creating the training target variable(70% of target variable)

def getTrainingTarget(rawTraining,TrainingPercent = 70): 
    TrainingLen = math.ceil(len(rawTraining)*(TrainingPercent*0.01))
    t = rawTraining[:TrainingLen]
    return t 


# Making the Training dataMatrix which is 70% of input dataset

def TrainingDataMatrix(rawData, TrainingPercent = 70): 
    T_len = math.ceil(len(rawData[0])*0.01*TrainingPercent)
    d2 = rawData[:,0:T_len]
    return d2 


# Making the Validation target(15% of input dataset)

def getValTargetVector(rawData, ValPercent, TrainingCount): 
    valSize = math.ceil(len(rawData)*ValPercent*0.01)
    V_End = TrainingCount + valSize
    t =rawData[TrainingCount+1:V_End]
    return t


# Creating Validation dataMatrix(15% of input dataset)

def ValDataMatrix(rawData, ValPercent, TrainingCount): 
    valSize = math.ceil(len(rawData[0])*ValPercent*0.01)
    V_End = TrainingCount + valSize
    dataMatrix = rawData[:,TrainingCount+1:V_End]  
    return dataMatrix 
def getBigSigma(Data,TrainingPercent):
    BigSigma    = np.zeros((len(Data),len(Data)))
    DataT       = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))        
    varVect     = []
    for i in range(0,len(DataT[0])):
        vct = []
        for j in range(0,TrainingLen):
            vct.append(Data[i][j])    
        varVect.append(np.var(vct))
    
    for j in range(len(Data)):
        BigSigma[j][j] = varVect[j]  # BigSigma is getting assigned diagonally by all values of varVect
        
    
    return BigSigma
    
def getScalar(DataRow,Mu, BigSigInv):  
    R = np.subtract(DataRow,Mu)
    T = np.dot(BigSigInv,np.transpose(R))  
    L = np.dot(R,T)
    return L

def getRadialBasisOut(DataRow,Mu, BigSigInv):    
    phi_x = math.exp(-0.5*getScalar(DataRow,Mu,BigSigInv))
    return phi_x

def getPhiMatrix(Data, Mu, BigSigma, TrainingPercent = 70):
    DataT = np.transpose(Data) 
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))        
    PHI = np.zeros((TrainingLen,len(Mu))) 
    BigSigInv = np.linalg.inv(BigSigma) 
    for  C in range(len(Mu)):
        for R in range(TrainingLen):
            PHI[R][C] = getRadialBasisOut(DataT[R], Mu[C], BigSigInv)
    
    return PHI
    
def getWeightsClosedForm(PHI, T, Lambda):
    Lambda_I = np.identity(len(PHI[0]))
    for i in range(0,len(PHI[0])):
        Lambda_I[i][i] = Lambda
    PHI_T       = np.transpose(PHI)
    PHI_SQR     = np.dot(PHI_T,PHI)
    PHI_SQR_LI  = np.add(Lambda_I,PHI_SQR)
    PHI_SQR_INV = np.linalg.inv(PHI_SQR_LI)
    INTER       = np.dot(PHI_SQR_INV, PHI_T)
    W           = np.dot(INTER, T)
    return W 

def getValTest(VAL_PHI,W):
    Y = np.dot(W,np.transpose(VAL_PHI))
    return Y


# rms = root mean square

def getErms(VAL_TEST_OUT,ValDataAct): 
    sum = 0.0
    accuracy = 0.0
    counter = 0
    for i in range (0,len(VAL_TEST_OUT)):
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2)
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):
            counter+=1
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))
    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT))))

# Preparing Dataset
# For concatenatedFeatures Dataset

RawTargetConcat = getTargetVector(ConcatenatedFeatures)
RawDataConcat   = rawDataMatrix(ConcatenatedFeatures)


# For subtractedFeatures Dataset 

RawTargetSubtract = getTargetVector(subtractedFeatures)
RawDataSubtract   = rawDataMatrix(subtractedFeatures)

# Preparing Training Data
# For concatenatedFeatures Dataset

TrainingPercent=70
TrainingTargetConcat = getTrainingTarget(RawTargetConcat,TrainingPercent)
TrainingDataConcat   = TrainingDataMatrix(RawDataConcat,TrainingPercent)
print(TrainingTargetConcat.shape)
print(TrainingDataConcat.shape)

print()
# For subtractedFeatures Dataset 

TrainingTargetSubtract = getTrainingTarget(RawTargetSubtract,TrainingPercent)
TrainingDataSubtract   = TrainingDataMatrix(RawDataSubtract,TrainingPercent)
print(TrainingTargetSubtract.shape)
print(TrainingDataSubtract.shape)

# Preparing Validation Data
# For concatenatedFeatures Dataset

ValidationPercent= 15
ValDataActConcat = getValTargetVector(RawTargetConcat,ValidationPercent, (len(TrainingTargetConcat)))
ValDataConcat    = ValDataMatrix(RawDataConcat,ValidationPercent, (len(TrainingTargetConcat)))
print(ValDataActConcat.shape)
print(ValDataConcat.shape)

print()
# For subtractedFeatures Dataset 

ValDataActSubtract = getValTargetVector(RawTargetSubtract,ValidationPercent, (len(TrainingTargetSubtract)))
ValDataSubtract   = ValDataMatrix(RawDataSubtract,ValidationPercent, (len(TrainingTargetSubtract)))
print(ValDataActSubtract.shape)
print(ValDataSubtract.shape)

# Preparing Test Data
# For concatenatedFeatures Dataset

TestPercent = 15
TestDataActConcat = getValTargetVector(RawTargetConcat,TestPercent, (len(TrainingTargetConcat)+len(ValDataActConcat)))
TestDataConcat = ValDataMatrix(RawDataConcat,TestPercent, (len(TrainingTargetConcat)+len(ValDataActConcat)))
print(TestDataActConcat.shape)
print(TestDataConcat.shape)

print()
# For subtractedFeatures Dataset 

TestDataActSubtract = getValTargetVector(RawTargetSubtract,TestPercent, (len(TrainingTargetSubtract)+len(ValDataActSubtract)))
TestDataSubtract = ValDataMatrix(RawDataSubtract,TestPercent, (len(TrainingTargetSubtract)+len(ValDataActSubtract)))
print(TestDataActSubtract.shape)
print(TestDataSubtract.shape)

# For concatenatedFeatures Dataset

k_list = [2*M for M in range(1,23)]
WCSS = [] # Within cluster sum of square
for M in k_list:
    kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(TrainingDataConcat))
    WCSS.append(kmeans.inertia_)

# Elbow Method
plt.plot(k_list,WCSS,'o-')
plt.xlabel("Number of clusters M----->")
plt.ylabel("Within cluster sum of square----->")
plt.title("WCSS Vs. Number of clusters M")

# For subtractedFeatures Dataset

k_list = [2*M for M in range(1,23)]
WCSS = [] # Within cluster sum of square
for M in k_list:
    kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(TrainingDataSubtract))
    WCSS.append(kmeans.inertia_)

# Elbow Method
plt.plot(k_list,WCSS,'o-')
plt.xlabel("Number of clusters M----->")
plt.ylabel("Within cluster sum of square----->")
plt.title("WCSS Vs. Number of clusters M")

# Closed form Solution
C_Lambda = 0.3
TrainingPercent = 70
ValidationPercent = 15
TestPercent = 15


# For concatenatedFeatures Dataset

kmeans = KMeans(n_clusters=9, random_state=0).fit(np.transpose(TrainingDataConcat))

Mu_c = kmeans.cluster_centers_
BigSigmaConcat      = getBigSigma(RawDataConcat, TrainingPercent)
TRAINING_PHI_Concat = getPhiMatrix(RawDataConcat, Mu_c, BigSigmaConcat, TrainingPercent)
W_Concat            = getWeightsClosedForm(TRAINING_PHI_Concat,TrainingTargetConcat,(C_Lambda)) 
TEST_PHI_Concat     = getPhiMatrix(TestDataConcat, Mu_c, BigSigmaConcat,100) 
VAL_PHI_Concat      = getPhiMatrix(ValDataConcat, Mu_c, BigSigmaConcat,100)


# For subtractedFeatures Dataset 
C_Lambda_ = 2.5
kmeans_ = KMeans(n_clusters=7, random_state=0).fit(np.transpose(TrainingDataSubtract))

Mu = kmeans_.cluster_centers_
BigSigmaSubtract      = getBigSigma(RawDataSubtract, TrainingPercent)
TRAINING_PHI_Subtract = getPhiMatrix(RawDataSubtract, Mu, BigSigmaSubtract, TrainingPercent)
W_Subtract            = getWeightsClosedForm(TRAINING_PHI_Subtract,TrainingTargetSubtract,(C_Lambda_)) 
TEST_PHI_Subtract     = getPhiMatrix(TestDataSubtract, Mu, BigSigmaSubtract,100) 
VAL_PHI_Subtract      = getPhiMatrix(ValDataSubtract, Mu, BigSigmaSubtract,100)

# For concatenatedFeatures Dataset

print(Mu_c.shape)
print(BigSigmaConcat.shape)
print(TRAINING_PHI_Concat.shape)
print(W_Concat.shape)
print(VAL_PHI_Concat.shape)
print(TEST_PHI_Concat.shape)
print()

# For subtractedFeatures Dataset 

print(Mu.shape)
print(BigSigmaSubtract.shape)
print(TRAINING_PHI_Subtract.shape)
print(W_Subtract.shape)
print(VAL_PHI_Subtract.shape)
print(TEST_PHI_Subtract.shape)

# To choose a value of regularization parameter(Lamda) with COncatenated training set

Lamda=[lamda*0.42 for lamda in range(1,21)]
Training_Accuracy = []
for lamda in Lamda:
  Weight = getWeightsClosedForm(TRAINING_PHI_Concat,TrainingTargetConcat,lamda) 
  TR_TEST_OUT  = getValTest(TRAINING_PHI_Concat,Weight)
  TrainingAcc= str(getErms(TR_TEST_OUT,TrainingTargetConcat))
  Training_Accuracy.append(float(TrainingAcc.split(',')[0]))
plt.plot(Lamda,Training_Accuracy,'ro-')
plt.ylabel("Training_Accuracy")
plt.xlabel("Lamda")
plt.title("Training Accuracy Vs. Lamda")

# To choose a value of regularization parameter(Lamda) with Subtracted training set

Lamda=[lamda*0.35 for lamda in range(1,21)]
Training_Accuracy = []
for lamda in Lamda:
  Weight = getWeightsClosedForm(TRAINING_PHI_Subtract,TrainingTargetSubtract,lamda) 
  TR_TEST_OUT  = getValTest(TRAINING_PHI_Subtract,Weight)
  TrainingAcc= str(getErms(TR_TEST_OUT,TrainingTargetSubtract))
  Training_Accuracy.append(float(TrainingAcc.split(',')[0]))
plt.plot(Lamda,Training_Accuracy,'ro-')
plt.ylabel("Training_Accuracy")
plt.xlabel("Lamda")
plt.title("Training Accuracy Vs. Lamda")

# Finding Erms on training, validation and test set
# For concatenatedFeatures Dataset

TR_TEST_OUT_Concat  = getValTest(TRAINING_PHI_Concat,W_Concat)
VAL_TEST_OUT_Concat = getValTest(VAL_PHI_Concat,W_Concat)
TEST_OUT_Concat     = getValTest(TEST_PHI_Concat,W_Concat)

TrainingAccuracy_Concat   = str(getErms(TR_TEST_OUT_Concat,TrainingTargetConcat))
ValidationAccuracy_Concat = str(getErms(VAL_TEST_OUT_Concat,ValDataActConcat))
TestAccuracy_Concat       = str(getErms(TEST_OUT_Concat,TestDataActConcat))

# For subtractedFeatures Dataset 

TR_TEST_OUT_Subtract  = getValTest(TRAINING_PHI_Subtract,W_Subtract)
VAL_TEST_OUT_Subtract = getValTest(VAL_PHI_Subtract,W_Subtract)
TEST_OUT_Subtract     = getValTest(TEST_PHI_Subtract,W_Subtract)

TrainingAccuracy_Subtract   = str(getErms(TR_TEST_OUT_Subtract,TrainingTargetSubtract))
ValidationAccuracy_Subtract = str(getErms(VAL_TEST_OUT_Subtract,ValDataActSubtract))
TestAccuracy_Subtract      = str(getErms(TEST_OUT_Subtract,TestDataActSubtract))

# For concatenatedFeatures Dataset

print("# Accuracy of concatenatedFeatures Dataset\n")

print("Training accuracy   = " + TrainingAccuracy_Concat.split(',')[0])
print("Validation accuracy = " +  ValidationAccuracy_Concat.split(',')[0])
print("Test accuracy       = " +  TestAccuracy_Concat.split(',')[0]+"\n")

print ("E_rms Training   = " + str(float(TrainingAccuracy_Concat.split(',')[1])))
print ("E_rms Validation = " + str(float(ValidationAccuracy_Concat.split(',')[1])))
print ("E_rms Testing    = " + str(float(TestAccuracy_Concat.split(',')[1]))+"\n\n\n")


# For subtractedFeatures Dataset 

print("# Accuracy of subtractedFeatures Dataset\n")

print("Training accuracy   = " + TrainingAccuracy_Subtract.split(',')[0])
print("Validation accuracy = " +  ValidationAccuracy_Subtract.split(',')[0])
print("Test accuracy       = " +  TestAccuracy_Subtract.split(',')[0]+"\n")

print ("E_rms Training   = " + str(float(TrainingAccuracy_Subtract.split(',')[1])))
print ("E_rms Validation = " + str(float(ValidationAccuracy_Subtract.split(',')[1])))
print ("E_rms Testing    = " + str(float(TestAccuracy_Subtract.split(',')[1])))

# Gradient Descent Solution

# For Concatenated features dataset

W_Now        = np.dot(220, W_Concat) # It intializes with random value
La           = 2 # Lamda
learningRate = 0.125
L_Erms_Val   = []
L_Erms_TR    = []
L_Erms_Test  = []
W_Mat        = []

for i in range(0,400):
    
    #print ('---------Iteration: ' + str(i) + '--------------')
    Delta_E_D     = -np.dot((TrainingTargetConcat[i] - np.dot(np.transpose(W_Now),TRAINING_PHI_Concat[i])),TRAINING_PHI_Concat[i])
    La_Delta_E_W  = np.dot(La,W_Now)
    Delta_E       = np.add(Delta_E_D,La_Delta_E_W)    
    Delta_W       = -np.dot(learningRate,Delta_E)
    W_T_Next      = W_Now + Delta_W
    W_Now         = W_T_Next
    
    #-----------------TrainingData Accuracy---------------------#
    TR_TEST_OUT   = getValTest(TRAINING_PHI_Concat,W_T_Next) 
    Erms_TR       = getErms(TR_TEST_OUT,TrainingTargetConcat)
    L_Erms_TR.append(float(Erms_TR.split(',')[1]))
    
    #-----------------ValidationData Accuracy---------------------#
    VAL_TEST_OUT  = getValTest(VAL_PHI_Concat,W_T_Next) 
    Erms_Val      = getErms(VAL_TEST_OUT,ValDataActConcat)
    L_Erms_Val.append(float(Erms_Val.split(',')[1]))
    
    #-----------------TestingData Accuracy---------------------#
    TEST_OUT      = getValTest(TEST_PHI_Concat,W_T_Next) 
    Erms_Test = getErms(TEST_OUT,TestDataActConcat)
    L_Erms_Test.append(float(Erms_Test.split(',')[1]))
    
# To find the learning rate for Concatenated features

Learningrate = [0.009*i for i in range(1,21)]
Training_Erms=[]
for l in Learningrate:
    
    La_Delta_E_W  = np.dot(La,W_Now)
    Delta_E       = np.add(Delta_E_D,La_Delta_E_W)    
    Delta_W       = -np.dot(l,Delta_E)
    W_T_Next      = W_Now + Delta_W
    W_Now         = W_T_Next
    
    TR_TEST_OUT   = getValTest(TRAINING_PHI_Concat,W_T_Next) 
    Erms_TR       = getErms(TR_TEST_OUT,TrainingTargetConcat)
    Training_Erms.append(float(Erms_TR.split(',')[1]))

    
plt.plot(Learningrate,Training_Erms,'ro-')
plt.ylabel("Training_Erms")
plt.xlabel("Learningrate")
plt.title("Training_Erms Vs. Learningrate")

print ('----------Gradient Descent Solution for Concatenated features--------------------')
print("Accuracy")
print("Training accuracy   = " +TrainingAccuracy_Concat.split(',')[0])
print("Validation accuracy = " + ValidationAccuracy_Concat.split(',')[0])
print("Test accuracy       = " + TestAccuracy_Concat.split(',')[0]+"\n")
print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))

# For Subtracted features dataset

W_Now_       = np.dot(220, W_Subtract) # It intializes with random value
La           = 2 # Lamda
learningRate = 0.01
L_Erms_Val   = []
L_Erms_TR    = []
L_Erms_Test  = []
W_Mat        = []

for i in range(0,400):
    
    #print ('---------Iteration: ' + str(i) + '--------------')
    Delta_E_D     = -np.dot((TrainingTargetSubtract[i] - np.dot(np.transpose(W_Now_),TRAINING_PHI_Subtract[i])),TRAINING_PHI_Subtract[i])
    La_Delta_E_W  = np.dot(La,W_Now_)
    Delta_E       = np.add(Delta_E_D,La_Delta_E_W)    
    Delta_W       = -np.dot(learningRate,Delta_E)
    W_T_Next      = W_Now_ + Delta_W
    W_Now_        = W_T_Next
    
    #-----------------TrainingData Accuracy---------------------#
    TR_TEST_OUT   = getValTest(TRAINING_PHI_Subtract,W_T_Next) 
    Erms_TR       = getErms(TR_TEST_OUT,TrainingTargetSubtract)
    L_Erms_TR.append(float(Erms_TR.split(',')[1]))
    
    #-----------------ValidationData Accuracy---------------------#
    VAL_TEST_OUT  = getValTest(VAL_PHI_Subtract,W_T_Next) 
    Erms_Val      = getErms(VAL_TEST_OUT,ValDataActSubtract)
    L_Erms_Val.append(float(Erms_Val.split(',')[1]))
    
    #-----------------TestingData Accuracy---------------------#
    TEST_OUT      = getValTest(TEST_PHI_Subtract,W_T_Next) 
    Erms_Test = getErms(TEST_OUT,TestDataActSubtract)
    L_Erms_Test.append(float(Erms_Test.split(',')[1]))
    
# To find the learning rate for Subtracted features

Learningrate = [0.009*i for i in range(1,19)]
Training_Erms=[]
for l in Learningrate:
    
    La_Delta_E_W  = np.dot(La,W_Now_)
    Delta_E       = np.add(Delta_E_D,La_Delta_E_W)    
    Delta_W       = -np.dot(l,Delta_E)
    W_T_Next      = W_Now_ + Delta_W
    W_Now         = W_T_Next
    
    TR_TEST_OUT   = getValTest(TRAINING_PHI_Subtract,W_T_Next) 
    Erms_TR       = getErms(TR_TEST_OUT,TrainingTargetSubtract)
    Training_Erms.append(float(Erms_TR.split(',')[1]))

    
plt.plot(Learningrate,Training_Erms,'ro-')
plt.ylabel("Training_Erms")
plt.xlabel("Learningrate")
plt.title("Training_Erms Vs. Learningrate")

print ('----------Gradient Descent Solution for Subtracted features--------------------')
print("Accuracy")
print("Training accuracy   = " +TrainingAccuracy_Subtract.split(',')[0])
print("Validation accuracy = " + ValidationAccuracy_Subtract.split(',')[0])
print("Test accuracy       = " + TestAccuracy_Subtract.split(',')[0]+"\n")
print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))

# Logistic Regression

# We have our desirable dataset ConcatenaatedFeatures and SubtractedFearures
ConcatenatedFeatures.head()
subtractedFeatures.head()
Bias = pd.DataFrame(np.ones((len(ConcatenatedFeatures),1)), columns = ['Bias'])
Bias.head()
Bias.shape

# For ConccatenatedFeatures

ConcatenatedFeatures_ = pd.concat([Bias, ConcatenatedFeatures[ConcatenatedFeatures.columns[2:-1]]], axis=1)
ConcatenatedFeatures_.shape

ConcatenatedFeatures_.head()

# For SubtractedFeatures
subtractedFeatures_ = pd.concat([Bias, subtractedFeatures[subtractedFeatures.columns[2:-1]]], axis=1)
subtractedFeatures_.shape
subtractedFeatures_.head()

# For concatenatedfeatures
weight = np.zeros(19)  # Bceause 19 is the number of features including bias

def sigmoidFunction(inputData, weight):
    sigmoid = 1/(np.exp(-1 * np.matmul(inputData, weight)))
    return sigmoid
    
def cost_function(m,data,weight,Target):
    J = 1/m*(-1*Target.T.dot(np.log(sigmoidFunction(data,weight))) - (1 - Target).T.dot(np.log(1-sigmoidFunction(data,weight))))
    return J
    
def gd(m,weight,X, Target, Learningrate): # gd stands for Gradient Descent
    cost_functionList = [0]* 500 
    for i in tqdm_notebook(range(500)):
        weight = weight - (Learningrate/m) * ( X.T.dot((sigmoidFunction(X,weight) - Target)))
        cost = cost_function(m,X,weight,Target)
        cost_functionList[i] = cost
    return weight, cost_functionList
    
# For ConcatenatedFeatures training set
m = len(ConcatenatedFeatures_) # number of training examples
Target_ = ConcatenatedFeatures['target'].values.astype(float)
TrainingPercent = int((80/100)*len(Target_))
Target = Target_[:TrainingPercent]
Learningrate = 0.002
X_ = ConcatenatedFeatures_.values.astype(float)
concatInput = X_[:TrainingPercent]
New_weight_concat, costList_concat = gd(m,weight,concatInput,Target,Learningrate)

# For subtractedFeatures

weight_subtract = np.zeros(10)  # Bceause 10 is the number of features including bias

# For subtractedFeatures training set

Learningrate_ = 0.05
X_subtract = subtractedFeatures_.values.astype(float)
subtractInput = X_subtract[:TrainingPercent]
New_weight_subtract, costList_subtract = gd(m,weight_subtract,subtractInput,Target,Learningrate_)

# For concatenated test set

predict_concat = sigmoidFunction(X_[TrainingPercent+1:],New_weight_concat)

# For subtracted test set

predict_subtract = sigmoidFunction(X_subtract[TrainingPercent+1:],New_weight_subtract)

def estimatedOutput(predict,Target):
    right = 0
    wrong = 0
    
    for i in range(len(Target)):
        if np.around(predict[i]) == Target[i]:
            right += 1
        else:
            wrong +=1
    return right, wrong
    
# For concatenatedfeatures
Target_concat = Target_[TrainingPercent+1:]
Right, Wrong = estimatedOutput(predict_concat,Target_concat)
print("Accuracy is " + str(Right/len(Target_concat)*100))
print("Error is " + str(Wrong/len(Target_concat)*100))


print()
# For subtractedfeatures

Target_subtract = Target_[TrainingPercent+1:]
Right, Wrong = estimatedOutput(predict_subtract,Target_subtract)
print("Accuracy is " + str(Right/len(Target_subtract)*100))
print("Error is " + str(Wrong/len(Target_subtract)*100))

# For Concatenated set

Iterations = list(range(500))
plt.plot(Iterations,costList_concat,'r-')
plt.ylabel("Cost funcion Concat ----->")
plt.xlabel("Iterations ----->")
plt.title("Cost funcion J Vs. Iterations")

# For Subtracted set

Iterations = list(range(500))
plt.plot(Iterations,costList_subtract,'r-')
plt.ylabel("Cost funcion subtract ----->")
plt.xlabel("Iterations ----->")
plt.title("Cost funcion J Vs. Iterations")

# Artificial Neural Network¶
ConcatenatedFeatures_.head()
subtractedFeatures_.head()
subtractedFeatures_.values[:,1:]

# Training the Model
from sklearn.model_selection import train_test_split
# For concatenated set
X_train_concat, X_test_concat, y_train_concat, y_test_concat = train_test_split(ConcatenatedFeatures_.values[:,1:], ConcatenatedFeatures['target'].values,test_size =0.2,random_state=0)
# For subtracted set
X_train_subtract, X_test_subtract, y_train_subtract, y_test_subtract = train_test_split(subtractedFeatures_.values[:,1:], subtractedFeatures['target'].values,test_size =0.2,random_state=0)

import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier_concat = Sequential()
classifier_subtract =Sequential()

# Adding the input layer and the first hidden layer
# For concatenated dataset

classifier_concat.add(Dense(units=64, kernel_initializer='uniform',activation='relu',input_dim=18))

# For subtracted dataset

classifier_subtract.add(Dense(units=64, kernel_initializer='uniform',activation='relu',input_dim=9))

# For concatenated dataset

classifier_concat.add(Dense(units=1, kernel_initializer='uniform',activation='sigmoid'))

# For subtracted dataset

classifier_subtract.add(Dense(units=1, kernel_initializer='uniform',activation='sigmoid'))

# Compiling the ANN
# For concatenated dataset

classifier_concat.compile(optimizer='sgd',loss ='binary_crossentropy', metrics=['accuracy'])

# For subtracted dataset

classifier_subtract.compile(optimizer='sgd',loss ='binary_crossentropy', metrics=['accuracy'])
# fitting the ANN to the Training set

# For concatenated Dataset
classifier_concat.fit(X_train_concat,y_train_concat,batch_size=24,epochs=100)

# fitting the ANN to the Training set

# For subtracted Dataset
classifier_subtract.fit(X_train_subtract,y_train_subtract,batch_size=24,epochs=100)

# For concatenated Dataset

y_pred_concat = classifier_concat.predict(X_test_concat)

# For subtracted Dataset

y_pred_subtract = classifier_subtract.predict(X_test_subtract)

y_pred_concat = (y_pred_concat > 0.5)
y_pred_concat =y_pred_concat.astype(int)
y_pred_concat = [y_pred_concat[i][0] for i in range(len(y_pred_subtract))]

y_pred_subtract = (y_pred_subtract > 0.5)
y_pred_subtract = y_pred_subtract.astype(int)
y_pred_subtract = [y_pred_subtract[i][0] for i in range(len(y_pred_subtract))]

def evaluation(predict,Target):
    right = 0
    wrong = 0
    
    for i in range(len(Target)):
        if predict[i] == Target[i]:
            right += 1
        else:
            wrong +=1
    return right, wrong
    
# For concatenatedfeatures

Right_concat, Wrong_concat = evaluation(y_pred_concat,y_test_concat)
print("Accuracy is " + str(Right_concat/len(y_test_concat)*100))
print("Error is " + str(Wrong_concat/len(y_test_concat)*100))

print()
# For subtractedfeatures

Right_subtract, Wrong_subtract = evaluation(y_pred_subtract,y_test_subtract)
print("Accuracy is " + str(Right_subtract/len(y_test_subtract)*100))
print("Error is " + str(Wrong_subtract/len(y_test_subtract)*100))
