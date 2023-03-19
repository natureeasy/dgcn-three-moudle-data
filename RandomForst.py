import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.io as scio
import h5py
from sklearn.model_selection import train_test_split
from sklearn import svm
from xgboost import XGBClassifier as XGBC
from sklearn.ensemble import GradientBoostingClassifier as GTBC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
import os
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")


def XGBC_test(Feature_Matrix, Label, kfold,Features_Left_Ratio):
    model = XGBC(n_estimators=120, random_state=1, use_label_encoder=False, max_depth=3, learning_rate=0.1,
                 colsample_bytree=0.7)
    strtfdKFold = StratifiedKFold(n_splits=kfold, shuffle=True)
    scores = []
    num_features = int(Feature_Matrix.shape[1]*Features_Left_Ratio)
    k_fold = strtfdKFold.split(Feature_Matrix, Label)
    pridict_label = np.zeros((len(Label), 1))
    for k, (train, test) in enumerate(k_fold):
        train_outer = np.array(Feature_Matrix)[train, :]
        test_outer = np.array(Feature_Matrix)[test, :]
        label_outer = Label[train]
        index = np.arange(len(label_outer))
        index_features = np.arange(train_outer.shape[1])
        ## 特征选择 双样本t检验
        # index_NC = (index * label_outer).tolist()
        # index_NC = list(map(int, filter(lambda a: a > 0, index_NC)))
        # index_SZ = (index * (1 - label_outer)).tolist()
        # index_SZ = list(map(int, filter(lambda a: a > 0, index_SZ)))
        # train_outerNC = train_outer[index_NC, :]
        # train_outerSZ = train_outer[index_SZ, :]
        # t, pval = stats.ttest_ind(train_outerNC, train_outerSZ)
        # 特征选择阈值结果
        # # 0.1/train_outer.shape[0]  0.820 +/- 0.076
        # # 0.1/train_outer.shape[1]  0.798 +/- 0.113
        # # 0.1    0.872 +/- 0.057
        # p_label = np.int64(pval < 0.1)
        # feature_index = index_features * p_label
        # feature_index = list(filter(lambda a: a > 0, feature_index))
        # train_selected = train_outer[:, feature_index]
        # test_selected = test_outer[:, feature_index]
        model.fit(train_outer, label_outer)
        importances_ = model.feature_importances_
        importances = model.feature_importances_.tolist()
        indices = np.argsort(importances)[::-1].tolist()
        Selected_feature = {}
        Selected_feature_index = indices[:num_features]
        for j in range(num_features):
            Selected_feature[j] = importances[indices[j]]
        Selected_feature = pd.DataFrame(Selected_feature.items()).transpose()
        Selected_feature.columns = Selected_feature_index
        Selected_feature = Selected_feature.iloc[1, :]
        train_outer = pd.DataFrame(train_outer)
        label_outer = pd.DataFrame(label_outer)
        test_outer = pd.DataFrame(test_outer)
        X_train = train_outer.loc[:, Selected_feature_index]
        new_index = {}
        for i in range(num_features):
            new_index[i] = i
        X_train.columns = new_index
        X_train = X_train.values

        X_test = test_outer.loc[:, Selected_feature_index]
        X_test.columns = new_index
        X_test = X_test.values
        ##网格寻优
        # model_val = XGBC()
        # grid = GridSearchCV(model_val, rf_params, cv=5, scoring='accuracy')
        # grid.fit(train_selected, label_outer)
        # print(grid.best_params_)
        # PAR = grid.best_params_
        # ##test
        # model = XGBC(n_estimators=PAR['n_estimators'], max_depth=PAR['max_depth'], colsample_bytree=PAR['colsample_bytree'], learning_rate = PAR['learning_rate'])
        model_new = XGBC(n_estimators=120, random_state=1, use_label_encoder=False, max_depth=3, learning_rate=0.1,
                     colsample_bytree=0.7)
        model_new.fit(X_train, label_outer)
        # score = model.score(X_test, Label[test])
        # scores.append(score)

        pridict_label[test, :] = model_new.predict(X_test).reshape(test_outer.shape[0], 1)
        # scores.append(score)
    acc = metrics.accuracy_score(Label, pridict_label)
    score = metrics.precision_score(Label, pridict_label, average='macro')
    f1 = metrics.f1_score(Label, pridict_label, average='weighted')
    recall = metrics.recall_score(Label, pridict_label, average='macro')
    # print('Fold: %2d,feature: %2d, Accuracy: %.3f' % (k+1, num_features, score))
    # print('\n\nCross-Validation accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))
    print('acc: %.3f score: %.3f f1: %.3f recall: %.3f' % (acc, score, f1, recall))
    return acc, score, f1, recall


if __name__ == '__main__':

    FC_path = r'I:\work\dgcn\all_module'
    result_root = r'I:\work\dgcn\all_module\all_mudule3000_PCArate'
    label_path = r'I:\work\dgcn\label.mat'
    Label = scio.loadmat(label_path)['label'].reshape(-1)
    # rf_params = {
    #     'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
    #     "kernel": ['linear', 'poly', 'rbf', 'sigmoid']
    # }
    num_features = np.arange(0.90, 1.00, 0.01)
    Label = np.int64(Label > 0)
    # Result_root = os.path.join(result_root, ("feature_" + str(num_features[i])))
    Accs = np.zeros((5, len(num_features)))
    Scores = np.zeros((5, len(num_features)))
    F1s = np.zeros((5, len(num_features)))
    Recalls = np.zeros((5, len(num_features)))
    j = 0
    for root, dirs, files in os.walk(FC_path):
        for file in files:
            # FC DTI GMV
            path = os.path.join(root, file)
            feature = scio.loadmat(path)
            feature = feature[list(feature.keys())[-1]]
            # 归一化
            feature_mean = np.mean(feature, 1).reshape(-1, 1)
            feature_std = np.std(feature, 1).reshape(-1, 1)
            feature_n = (feature - feature_mean) / feature_std
            if j == 0:
                feature_final = feature_n
            else:
                feature_final = np.concatenate((feature_final, feature_n), axis=1)
            j = j + 1
    Feature_Matrix = feature_final;
    scores_result = np.zeros((5, len(num_features)))
    for i in range(len(num_features)):

        # result_path = os.path.join(Result_root, file[:-4])

        for k in range(5):
            # acc, score, f1, recall = SVM_test(Feature_Matrix, Label, 10, rf_params, num_features[i], '2ttest')
            acc, score, f1, recall = XGBC_test(Feature_Matrix, Label, 10, num_features[i])
            Accs[k][i] = acc
            Scores[k][i] = score
            F1s[k][i] = f1
            Recalls[k][i] = recall
            # CC = np.mean(feature_indexs,0)
            # CC = np.int64(CC == 1.0)
            # index_features = np.arange(Feature_Matrix.shape[1])
            # CC = index_features*CC
            # index_CC = list(filter(lambda a:a>0, CC))
            # scores_result[k,i] = scores_mean
            print('Loop: %2d, feature: %2d' % (k + 1, num_features[i]))
    Accs_final = np.mean(Accs, axis=0).reshape(1, -1)
    Scores_final = np.mean(Scores, axis=0).reshape(1, -1)
    F1s_final = np.mean(F1s, axis=0).reshape(1, -1)
    Recalls_final = np.mean(Recalls, axis=0).reshape(1, -1)
    final_result = np.concatenate((Accs_final, Scores_final, F1s_final, Recalls_final), axis=0)
    # file=open(scores_path,'w')
    # for m in range(len(scores_final)):
    #     file.writelines((str(scores_final[m])+"\n"));
    # file.close()
    # result_root = 'F:/Documents/Desktop/SvmScript/SVM/python/final_result/all_nodule_new'
    acc_path = os.path.join(result_root, 'final.txt')
    file = open(acc_path, 'w')
    for i in range(len(final_result)):
        file.writelines((str(final_result[i]) + "\n"));
    file.close()