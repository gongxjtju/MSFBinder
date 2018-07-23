import random

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def spiltdata(feature, label, i, testspilt):
    x_test = feature[int(i * len(feature) * testspilt):int((i + 1) * len(feature) * testspilt)]
    y_test = label[int(i * len(label) * testspilt):int((i + 1) * len(label) * testspilt)]
    x_train = []
    y_train = []
    if int(i * len(feature) * testspilt) != 0 and int((i + 1) * len(feature) * testspilt) != (len(feature) - 1):
        x_train.extend(feature[:int(i * len(feature) * testspilt)])
        x_train.extend(feature[int((i + 1) * len(feature) * testspilt):])
        y_train.extend(label[:int(i * len(label) * testspilt)])
        y_train.extend(label[int((i + 1) * len(label) * testspilt):])
    if int((i + 1) * len(feature) * testspilt) == (len(feature) - 1):
        x_train.extend(feature[:int(i * len(feature) * testspilt)])
        y_train.extend(label[:int(i * len(label) * testspilt)])
    if int(i * len(feature) * testspilt) == 0:
        x_train.extend(feature[int((i + 1) * len(feature) * testspilt):])
        y_train.extend(label[int((i + 1) * len(label) * testspilt):])
    return (x_test, y_test), (x_train, y_train)
def yizhe(feature_188D, label_188D, feature_localdpp, label_localdpp, feature_acstruct, label_acstruct,
          feature_dwt, label_dwt, i, testspilt,classifier):
    (x_test_188D, y_test_188D), (x_train_188D, y_train_188D) = spiltdata(feature_188D, label_188D, i, testspilt)
    (x_test_localdpp, y_test_localdpp), (x_train_localdpp, y_train_localdpp) = spiltdata(feature_localdpp, label_localdpp, i,
                                                                                     testspilt)
    (x_test_acstruct, y_test_acstruct), (x_train_acstruct, y_train_acstruct) = spiltdata(feature_acstruct,
                                                                                                 label_acstruct, i,
                                                                                                 testspilt)
    (x_test_dwt, y_test_dwt), (x_train_dwt, y_train_dwt) = spiltdata(feature_dwt, label_dwt, i,
                                                                                 testspilt)
    if(classifier=='SVM'):
        model_localdpp = SVC(C=2.0, gamma=0.0625, probability=True)
    elif(classifier=='LR'):
        model_localdpp = LogisticRegression(solver='lbfgs')
    else:
        model_localdpp = RandomForestClassifier(n_estimators=290, random_state=0)
    model_localdpp.fit(x_train_localdpp, y_train_localdpp)
    pred_localdpp = model_localdpp.predict_proba(x_test_localdpp)
    pred_localdpp1 = pred_localdpp[:, 1]
    if(classifier=='SVM'):
        model_188D = SVC(C=1.0, gamma=0.125, probability=True)
    elif(classifier=='LR'):
        model_188D=LogisticRegression(solver='lbfgs')
    else:
        model_188D = RandomForestClassifier(random_state=0, n_estimators=490)
    model_188D.fit(x_train_188D, y_train_188D)
    pred_188D = model_188D.predict_proba(x_test_188D)
    pred_188D1 = pred_188D[:, 1]
    if(classifier=="SVM"):
        model_dwt = SVC(C=2.0, gamma=0.0625, probability=True)
    elif(classifier=='LR'):
        model_dwt=LogisticRegression(solver='lbfgs')
    else:
        model_dwt = RandomForestClassifier(n_estimators=660, random_state=0)
    model_dwt.fit(x_train_dwt, y_train_dwt)
    pred_dwt = model_dwt.predict_proba(x_test_dwt)
    pred_dwt1 = pred_dwt[:, 1]
    if(classifier=="SVM"):
        model_acstruct = SVC(C=16.0, gamma=0.25, probability=True)
    elif(classifier=="LR"):
        model_acstruct=LogisticRegression(solver='lbfgs')
    else:
        model_acstruct = RandomForestClassifier(n_estimators=190, random_state=0)
    model_acstruct.fit(x_train_acstruct, y_train_acstruct)
    pred_acstruct = model_acstruct.predict_proba(x_test_acstruct)
    pred_acstruct1 = pred_acstruct[:, 1]
    pred = np.vstack((pred_localdpp1, pred_188D1, pred_dwt1,
                      pred_acstruct1)).T
    test_pred=np.mean(pred,axis=1)
    # print test_pred.shape
    for i in range(len(test_pred)):
        if test_pred[i]>0.5:
            test_pred[i]=1
        else:
            test_pred[i]=0
    mcc = matthews_corrcoef(y_test_188D, test_pred)
    tn,fp,fn,tp=confusion_matrix(y_test_188D,test_pred).ravel()
    SN = tp*1.0 / (tp + fn)
    SP = tn*1.0 / (fp + tn)
    acc=accuracy_score(y_test_188D, test_pred)
    return (acc,mcc,SN,SP)

def voting(count,seed):
    feature_localdpp = np.loadtxt(open("./featuredata/g_feature_gai1221_local_2_2_guiyihua.csv"), delimiter=",",
                                 skiprows=0)
    label_localdpp = np.loadtxt(open("./featuredata/g_label_CT_1211.csv"), delimiter=",", skiprows=0)
    feature_dwt = np.loadtxt(open("./featuredata/PSSM_DWT_feature_guiyihua.csv"), delimiter=",",skiprows=0)
    label_dwt = np.loadtxt(open("./featuredata/g_label_CT_1211.csv"), delimiter=",", skiprows=0)
    feature_188D = np.loadtxt(open("./featuredata/188D_guiyihua.csv"), delimiter=",", skiprows=0)
    label_188D = np.loadtxt(open("./featuredata/g_label_CT_1211.csv"), delimiter=",", skiprows=0)
    feature_acstruct = np.loadtxt(open("./featuredata/g_feature_gai1221_structual_guiyihua.csv"),
                                        delimiter=",", skiprows=0)
    label_acstruct = np.loadtxt(open("./featuredata/g_label_CT_1211.csv"), delimiter=",",
                                      skiprows=0)
    np.random.seed(seed)
    np.random.shuffle(feature_localdpp)
    np.random.seed(seed)
    np.random.shuffle(label_localdpp)
    np.random.seed(seed)
    np.random.shuffle(feature_dwt)
    np.random.seed(seed)
    np.random.shuffle(label_dwt)
    np.random.seed(seed)
    np.random.shuffle(feature_188D)
    np.random.seed(seed)
    np.random.shuffle(label_188D)
    np.random.seed(seed)
    np.random.shuffle(feature_acstruct)
    np.random.seed(seed)
    np.random.shuffle(label_acstruct)
    accsum=0
    mccsum=0
    SNsum=0
    SPsum=0
    classifier='SVM'
    for i in range(5):
        # print i
        (acc,mcc,SN,SP)=yizhe(feature_188D, label_188D, feature_localdpp, label_localdpp, feature_acstruct, label_acstruct,feature_dwt, label_dwt, i, 0.2,classifier)
        accsum=accsum+acc
        mccsum=mccsum+mcc
        SNsum=SNsum+SN
        SPsum=SP+SPsum
    acc=accsum/5.0
    mcc=mccsum/5.0
    SN=SNsum/5.0
    SP=SPsum/5.0
    print "di ",count,"ci"
    print "acc:",acc
    print "SN:", SN
    print "SP:", SP
    print "MCC:", mcc
for i in range(5):
    seed=random.randint(1,100)
    voting(i+1,seed)
