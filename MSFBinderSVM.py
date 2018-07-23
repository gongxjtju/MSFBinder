import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
def spiltdata_liuyi(feature, label, i, testspilt):
    x_test=feature[i]
    x_test=np.array(x_test)
    x_test=x_test.reshape(1,-1)
    y_test=label[i]
    y_test=np.array(y_test)
    y_test=y_test.reshape(1,-1)
    if i==0:
       x_train=feature[1:]
       y_train=label[1:]
    elif i==(len(feature)-1):
        x_train=feature[:i]
        y_train=label[:i]
    else:
        x_train=[]
        y_train=[]
        x_train.extend(feature[:i])
        x_train.extend(feature[(i+1):])
        y_train.extend(label[:i])
        y_train.extend(label[(i+1):])
    return (x_test, y_test), (x_train, y_train)
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
    pred_localdpp = model.predict_proba(x_test)
    pred_localdpp1 = pred_localdpp[:, 1]
    return pred_localdpp1
def yizhe(feature_188D, label_188D, feature_localdpp, label_localdpp, feature_acstruct, label_acstruct,
          feature_dwt, label_dwt, i, testspilt,localdppparam,se188Dparam,dwtparam,acstructparam):
    (x_test_188D, y_test_188D), (x_train_188D, y_train_188D) = spiltdata_liuyi(feature_188D, label_188D, i, testspilt)
    (x_test_localdpp, y_test_localdpp), (x_train_localdpp, y_train_localdpp) = spiltdata_liuyi(feature_localdpp, label_localdpp, i,
                                                                                     testspilt)
    (x_test_acstruct, y_test_acstruct), (x_train_acstruct, y_train_acstruct) = spiltdata_liuyi(feature_acstruct,
                                                                                                 label_acstruct, i,
                                                                                                 testspilt)
    (x_test_dwt, y_test_dwt), (x_train_dwt, y_train_dwt) = spiltdata_liuyi(feature_dwt, label_dwt, i,
                                                                                 testspilt)
      #trining the first layer
	model_localdpp = SVC(C=2.0, gamma= 0.0625, probability=True)
    model_localdpp.fit(x_train_localdpp, y_train_localdpp)
    pred_localdpp = model_localdpp.predict_proba(x_test_localdpp)
    pred_localdpp1 = pred_localdpp[:, 1]
  
    model_188D = SVC(C=1.0, gamma=0.125, probability=True)
    model_188D.fit(x_train_188D, y_train_188D)
    pred_188D = model_188D.predict_proba(x_test_188D)
    pred_188D1 = pred_188D[:, 1]
    model_dwt = SVC(C=2.0, gamma=0.0625, probability=True)
    model_dwt.fit(x_train_dwt, y_train_dwt)
    pred_dwt = model_dwt.predict_proba(x_test_dwt)
    pred_dwt1 = pred_dwt[:, 1]
    model_acstruct = SVC(C=16.0, gamma=0.25, probability=True)
    model_acstruct.fit(x_train_acstruct, y_train_acstruct)
    pred_acstruct = model_acstruct.predict_proba(x_test_acstruct)
    pred_acstruct1 = pred_acstruct[:, 1]
    pred = np.vstack((pred_localdpp1,  pred_188D1,  pred_dwt1,
                       pred_acstruct1)).T
    print pred.shape
    # train the second layer
    y_pred_localdpp=[]
    y_pred_188D = []
    y_pred_acstruct = []
    y_pred_dwt = []
    for i in range(5):
        y_pred_localdpp.extend(yizhetwolayer(x_train_localdpp, y_train_localdpp, localdppparam, i, 0.2))
        y_pred_188D.extend(yizhetwolayer(x_train_188D, y_train_188D, se188Dparam, i, 0.2))
        y_pred_acstruct.extend(yizhetwolayer(x_train_acstruct, y_train_acstruct, acstructparam, i, 0.2))
        y_pred_dwt.extend( yizhetwolayer(x_train_dwt, y_train_dwt, dwtparam, i, 0.2))
    pred_zong = np.vstack((y_pred_localdpp, y_pred_188D, y_pred_dwt, y_pred_acstruct)).T
    print pred_zong.shape
    model=LogisticRegression(solver='lbfgs')
    model.fit(pred_zong, y_train_acstruct)
    test_pred = model.predict(pred)
    return (test_pred)

def MSFBinderSVM():
    feature_localdpp = np.loadtxt(open("./featuredata/g_feature_gai1221_local_2_2_guiyihua.csv"), delimiter=",",
                                 skiprows=0)
    label_localdpp = np.loadtxt(open("./featuredata/g_label_CT_1211.csv"), delimiter=",", skiprows=0)
    feature_dwt = np.loadtxt(open("./featuredata/PSSM_DWT_feature_guiyihua.csv"), delimiter=",",
                                skiprows=0)
    label_dwt = np.loadtxt(open("./featuredata/g_label_CT_1211.csv"), delimiter=",", skiprows=0)
    feature_188D = np.loadtxt(open("./featuredata/188D_guiyihua.csv"), delimiter=",", skiprows=0)
    label_188D = np.loadtxt(open("./featuredata/g_label_CT_1211.csv"), delimiter=",", skiprows=0)
    feature_acstructquan = np.loadtxt(open("./featuredata/g_feature_gai1221_structual_guiyihua.csv"),
                                        delimiter=",", skiprows=0)
    label_acstructquan = np.loadtxt(open("./featuredata/g_label_CT_1211.csv"), delimiter=",",
                                      skiprows=0)
    localdppparam = np.loadtxt(open("./paramdata/two_local_2_2_param.csv"), delimiter=",", skiprows=0)
    se188Dparam = np.loadtxt(open("./paramdata/two_188D_param.csv"), delimiter=",", skiprows=0)
    dwtparam = np.loadtxt(open("./paramdata/two_DWT_param.csv"), delimiter=",", skiprows=0)
    acstructparam = np.loadtxt(open("./paramdata/two_struc_param.csv"), delimiter=",", skiprows=0)

    np.random.seed(12)
    np.random.shuffle(feature_localdpp)
    np.random.seed(12)
    np.random.shuffle(label_localdpp)
    np.random.seed(12)
    np.random.shuffle(feature_dwt)
    np.random.seed(12)
    np.random.shuffle(label_dwt)
    np.random.seed(12)
    np.random.shuffle(feature_188D)
    np.random.seed(12)
    np.random.shuffle(label_188D)
    np.random.seed(12)
    np.random.shuffle(feature_acstructquan)
    np.random.seed(12)
    np.random.shuffle(label_acstructquan)
    y_test=[]
    for i in range(len(feature_acstructquan)):
        print i
        temp=yizhe(feature_188D, label_188D, feature_localdpp, label_localdpp, feature_acstructquan, label_acstructquan,
          feature_dwt, label_dwt, i, 0.2,localdppparam,se188Dparam,dwtparam,acstructparam)
        y_test.extend(temp)
    print len(y_test)
    print(accuracy_score(label_188D, y_test))
    mcc = matthews_corrcoef(label_188D, y_test)
    tn, fp, fn, tp = confusion_matrix(label_188D, y_test).ravel()
    SN = tp * 1.0 / (tp + fn)
    SP = tn * 1.0 / (fp + tn)
    print "SN:", SN
    print "SP:", SP
    print "MCC:", mcc

MSFBinderSVM()
