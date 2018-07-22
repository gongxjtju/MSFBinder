import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import random

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
def yizhetwolayer(feature,label,param,i,testspilt):
    (x_test, y_test), (x_train, y_train) = spiltdata(feature, label, i, testspilt)
    model = SVC(C=param[i*2], gamma=param[i*2+1], probability=True)
    model.fit(x_train, y_train)
    pred_pssmphy = model.predict_proba(x_test)
    pred_pssmphy1 = pred_pssmphy[:, 1]
    return pred_pssmphy1
def yizhetwolayerrf(feature,label,param,i,testspilt):
    (x_test, y_test), (x_train, y_train) = spiltdata(feature, label, i, testspilt)
    model = RandomForestClassifier(n_estimators=int(param[i]),random_state=0)
    model.fit(x_train, y_train)
    pred_pssmphy = model.predict_proba(x_test)
    pred_pssmphy1 = pred_pssmphy[:, 1]
    return pred_pssmphy1
def yizhetwolayerpusu(feature,label,i,testspilt):
    (x_test, y_test), (x_train, y_train) = spiltdata(feature, label, i, testspilt)
    model = MultinomialNB(fit_prior=True)
    model.fit(x_train, y_train)
    pred_pssmphy = model.predict_proba(x_test)
    pred_pssmphy1 = pred_pssmphy[:, 1]
    return pred_pssmphy1
def yizhe(feature_188D, label_188D, feature_pssmphy, label_pssmphy, feature_huadongDIS, label_huadongDIS,
          feature_pseaac, label_pseaac, i, testspilt,pssmphyparam,se188Dparam,pseaacparam,huadongDISparam,pssmphyparamrf,se188Dparamrf,pseaacparamrf,huadongDISparamrf):
    (x_test_188D, y_test_188D), (x_train_188D, y_train_188D) = spiltdata(feature_188D, label_188D, i, testspilt)
    (x_test_pssmphy, y_test_pssmphy), (x_train_pssmphy, y_train_pssmphy) = spiltdata(feature_pssmphy, label_pssmphy, i,
                                                                                     testspilt)
    (x_test_huadongDIS, y_test_huadongDIS), (x_train_huadongDIS, y_train_huadongDIS) = spiltdata(feature_huadongDIS,
                                                                                                 label_huadongDIS, i,
                                                                                                 testspilt)
    (x_test_pseaac, y_test_pseaac), (x_train_pseaac, y_train_pseaac) = spiltdata(feature_pseaac, label_pseaac, i,
                                                                                 testspilt)
    model_pssmphy = SVC(C=0.25, gamma=0.35355339059327379, probability=True)
    model_pssmphy.fit(x_train_pssmphy, y_train_pssmphy)
    pred_pssmphy = model_pssmphy.predict_proba(x_test_pssmphy)
    pred_pssmphy1 = pred_pssmphy[:, 1]
    model_188D = SVC(C=1.0, gamma=0.125, probability=True)
    model_188D.fit(x_train_188D, y_train_188D)
    pred_188D = model_188D.predict_proba(x_test_188D)
    pred_188D1 = pred_188D[:, 1]
    model_pseaac = SVC(C=2.0, gamma=0.0625, probability=True)
    model_pseaac.fit(x_train_pseaac, y_train_pseaac)
    pred_pseaac = model_pseaac.predict_proba(x_test_pseaac)
    pred_pseaac1 = pred_pseaac[:, 1]
    model_huadongDIS = SVC(C=16.0, gamma=0.25, probability=True)
    model_huadongDIS.fit(x_train_huadongDIS, y_train_huadongDIS)
    pred_huadongDIS = model_huadongDIS.predict_proba(x_test_huadongDIS)
    pred_huadongDIS1 = pred_huadongDIS[:, 1]
    pred = np.vstack((pred_pssmphy1, pred_188D1, pred_pseaac1,
                      pred_huadongDIS1)).T
    print pred.shape
    y_pred_pssmphy=[]
    y_pred_188D = []
    y_pred_huadongDIS = []
    y_pred_pseaac = []
    j=i
    for i in range(5):
        y_pred_pssmphy.extend(yizhetwolayer(x_train_pssmphy, y_train_pssmphy,pssmphyparam,i,0.2))
        y_pred_188D.extend(yizhetwolayer(x_train_188D, y_train_188D,se188Dparam,i,0.2))
        y_pred_huadongDIS.extend(yizhetwolayer(x_train_huadongDIS, y_train_huadongDIS, huadongDISparam, i, 0.2))
        y_pred_pseaac.extend(yizhetwolayer(x_train_pseaac, y_train_pseaac, pseaacparam, i, 0.2))
    pred_zong = np.vstack((y_pred_pssmphy, y_pred_188D,y_pred_pseaac,
                                y_pred_huadongDIS)).T
    print pred_zong.shape
    model=LogisticRegression(solver='lbfgs')
    model.fit(pred_zong, y_train_huadongDIS)
    print model.coef_
    test_pred = model.predict(pred)
    print len(test_pred)
    mcc = matthews_corrcoef(y_test_188D, test_pred)
    tn,fp,fn,tp=confusion_matrix(y_test_188D,test_pred).ravel()
    SN = tp*1.0 / (tp + fn)
    SP = tn*1.0 / (fp + tn)
    acc=accuracy_score(y_test_188D, test_pred)
    print "SN:",SN
    print "SP:",SP
    print "MCC:",mcc
    return (acc,mcc,SN,SP)
