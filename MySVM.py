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


def XGBC_test(Feature_Matrix, Label, kfold, rf_params):
    model = XGBC(n_estimators=120, random_state=1, use_label_encoder=False, max_depth=3, learning_rate=0.1, colsample_bytree =0.7)
    strtfdKFold = StratifiedKFold(n_splits = kfold, shuffle = True)
    scores = []
    for k, (train, test) in enumerate(kfold):
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
        num_features = 50
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
        X_train= train_outer.loc[:, Selected_feature_index]
        new_index = {}
        for i in range(num_features):
            new_index[i] = i
        X_train.columns = new_index
        X_train = X_train.values

        X_test = test_outer.loc[:, Selected_feature_index]
        X_test.columns = new_index
        X_test= X_test.values
        ##网格寻优
        # model_val = XGBC()
        # grid = GridSearchCV(model_val, rf_params, cv=5, scoring='accuracy')
        # grid.fit(train_selected, label_outer)
        # print(grid.best_params_)
        # PAR = grid.best_params_
        # ##test
        # model = XGBC(n_estimators=PAR['n_estimators'], max_depth=PAR['max_depth'], colsample_bytree=PAR['colsample_bytree'], learning_rate = PAR['learning_rate'])
        model = XGBC(n_estimators=120, random_state=1, use_label_encoder=False, max_depth=3, learning_rate=0.1, colsample_bytree =0.7)
        model.fit(X_train, label_outer)
        score = model.score(X_test, Label[test])
        scores.append(score)
        print('Fold: %2d, Accuracy: %.3f' % (k + 1, score))

    print('\n\nCross-Validation accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

    return

def SVM_test(Feature_Matrix, Label, kfold, rf_params, num_features, feature_selection):
    strtfdKFold = StratifiedKFold(n_splits=10, shuffle=True)
    
    kfold = strtfdKFold.split(Feature_Matrix, Label)
    feature_indexs = np.zeros((10,Feature_Matrix.shape[1]))
    pridict_label = np.zeros((len(Label),1))
    for k, (train, test) in enumerate(kfold):
        train_outer = np.array(Feature_Matrix)[train,:]
        test_outer = np.array(Feature_Matrix)[test,:]
        label_outer = Label[train]
        index = np.arange(len(label_outer))
        index_features = np.arange(train_outer.shape[1])
        if feature_selection == '2ttest':
        ## 特征选择 双样本t检验
            index_NC = (index*label_outer).tolist()
            index_NC = list(filter(lambda a:a>0, index_NC))
            index_SZ = (index*(1-label_outer)).tolist()
            index_SZ = list(filter(lambda a:a>0, index_SZ))
            train_outerNC = train_outer[index_NC,:]
            train_outerSZ = train_outer[index_SZ,:]
            t, pval = stats.ttest_ind(train_outerNC, train_outerSZ) 
            tmp = np.array([pval,index_features])
            #x = tmp.T[np.lexsort(tmp[::-1,:])].T
            tmp = tmp.T[np.lexsort(tmp[::-1,:])].T
            turelabel = tmp[:][1].astype('int')
            num_features = int(num_features*len(pval))
            feature_index = turelabel[:num_features]
            #feature_index = turelabel[:num_features]   #original
            # 特征选择阈值结果
            # p_label = np.int64(pval<0.1)
            #feature_index = index_features*p_label
            #feature_index = list(filter(lambda a:a>0, feature_index))
            train_selected = train_outer[:,feature_index]
            test_selected = test_outer[:,feature_index]
            feature_indexs[k,feature_index] = 1
        elif feature_selection == 'PCA':           
            pca = RandomizedPCA(n_components=num_features, whiten=True).fit(train_outer)
            train_selected = pca.transform(train_outer)
            test_selected = pca.transform(test_outer)
        
        ##网格寻优
        model_val = svm.SVC(gamma='scale')
        grid = GridSearchCV(model_val, rf_params, cv=10, scoring='accuracy')
        grid.fit(train_selected, label_outer)
        #print(grid.best_params_)
        PAR = grid.best_params_
        ##test
        model = svm.SVC(C = PAR['C'],kernel = PAR['kernel'],gamma='scale')
        model.fit(train_selected, label_outer)
        pridict_label[test,:] = model.predict(test_selected).reshape(test_outer.shape[0],1)
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
    '''
     单个指标的svm结果
    '''
    # from sklearn.decomposition import PCA
    # pca = PCA()
    # pca.fit(Feature_Matrix)  #训练模型
    # pca.components_  #返回模型特征向量
    # ratio = pca.explained_variance_ratio_   #返回各个成分各子方差百分比
    # np.cumsum(ratio)  #累加
    # FC_path = r'F:\Documents\Desktop\SvmScript\DATA\WUXI\wuxi_172_FC_MC_SC\oriVector\all_module'
    # result_root = r'F:\Documents\Desktop\SvmScript\SVM\python\final_result\all_mudule3000_PCArate'
    # label_path = r'F:\Documents\Desktop\SvmScript\DATA\WUXI\wuxi_172_FC_MC_SC\label.mat'
    # Label = scio.loadmat(label_path)['label'].reshape(-1)
    # rf_params = {
    # 'C': [0.0001,0.001, 0.01,0.1,1,10,100,1000,10000],
    # "kernel":['linear','poly','rbf','sigmoid']
    # }
    # num_features = np.arange(0.90, 1.00, 0.01)
    # Label = np.int64(Label>0)
    #     # Result_root = os.path.join(result_root, ("feature_" + str(num_features[i])))
    # Accs = np.zeros((5, len(num_features)))
    # Scores = np.zeros((5, len(num_features)))
    # F1s = np.zeros((5, len(num_features)))
    # Recalls = np.zeros((5, len(num_features)))

    # for root, dirs, files in os.walk(FC_path):
    #     for file in files:
    #         path = os.path.join(root, file)
    #         result_path = os.path.join(result_root, file[:-4])
    #         if not os.path.exists(result_root):
    #             os.makedirs(result_root)
    #         if not os.path.exists(result_path):
    #             os.makedirs(result_path)
    #         feature_path = os.path.join(result_path,'features_index.txt')
    #         scores_path = os.path.join(result_path,'scores.txt')

    #         feature = scio.loadmat(path)
    #         feature = feature[list(feature.keys())[-1]]
    #         #归一化
    #         feature_mean = np.mean(feature,1).reshape(-1,1)
    #         feature_std = np.std(feature,1).reshape(-1,1)
    #         feature_n = (feature - feature_mean)/feature_std
    #         Feature_Matrix = pd.DataFrame(feature_n)
    #         scores_result = np.zeros((5,len(num_features)))
    #         for i in range(len(num_features)):

    #             # result_path = os.path.join(Result_root, file[:-4])

    #             for k in range(5):
    #                 acc, score, f1, recall = SVM_test(Feature_Matrix, Label, 10, rf_params, num_features[i], 'PCA')
    #                 Accs[k][i] = acc
    #                 Scores[k][i] = score
    #                 F1s[k][i] = f1
    #                 Recalls[k][i] = recall
    #                 #CC = np.mean(feature_indexs,0)
    #                 #CC = np.int64(CC == 1.0)
    #                 #index_features = np.arange(Feature_Matrix.shape[1])
    #                 #CC = index_features*CC
    #                 #index_CC = list(filter(lambda a:a>0, CC))
    #                 #scores_result[k,i] = scores_mean
    #                 print('Loop: %2d, feature: %2d' % (k + 1, num_features[i]))
    #         Accs_final = np.mean(Accs, axis = 0).reshape(1,-1)
    #         Scores_final = np.mean(Scores, axis = 0).reshape(1,-1)
    #         F1s_final = np.mean(F1s, axis = 0).reshape(1,-1)
    #         Recalls_final = np.mean(Recalls, axis = 0).reshape(1,-1)
    #         final_result = np.concatenate((Accs_final,Scores_final,F1s_final,Recalls_final), axis = 0)
    #         # file=open(scores_path,'w')
    #         # for m in range(len(scores_final)):
    #         #     file.writelines((str(scores_final[m])+"\n"));
    #         # file.close()
    #         # result_root = 'F:/Documents/Desktop/SvmScript/SVM/python/final_result/all_nodule_new'
    #         acc_path = os.path.join(result_path, 'final.txt')
    #         file=open(acc_path,'w')
    #         for i in range(len(final_result)):
    #             file.writelines((str(final_result[i])+"\n"));
    #         file.close()


    '''
     单个指标的svm结果
    '''
    # from sklearn.decomposition import PCA
    # pca = PCA()
    # pca.fit(Feature_Matrix)  #训练模型
    # pca.components_  #返回模型特征向量
    # ratio = pca.explained_variance_ratio_   #返回各个成分各子方差百分比
    # np.cumsum(ratio)  #累加
    FC_path = r'I:\work\dgcn\all_module'
    result_root = r'I:\work\dgcn\all_module\all_mudule3000_PCArate'
    label_path = r'I:\work\dgcn\label.mat'
    Label = scio.loadmat(label_path)['label'].reshape(-1)
    rf_params = {
    'C': [0.0001,0.001, 0.01,0.1,1,10,100,1000,10000],
    "kernel":['linear','poly','rbf','sigmoid']
    }
    num_features = np.arange(0.90, 1.00, 0.01)
    Label = np.int64(Label>0)
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
                #归一化
                feature_mean = np.mean(feature,1).reshape(-1,1)
                feature_std = np.std(feature,1).reshape(-1,1)
                feature_n = (feature - feature_mean)/feature_std
                if j == 0:
                    feature_final = feature_n
                else:
                    feature_final = np.concatenate((feature_final, feature_n), axis=1)
                j = j + 1
    Feature_Matrix = feature_final;
    scores_result = np.zeros((5,len(num_features)))
    for i in range(len(num_features)):
        
        # result_path = os.path.join(Result_root, file[:-4])
        
        for k in range(5):
            acc, score, f1, recall = SVM_test(Feature_Matrix, Label, 10, rf_params, num_features[i], '2ttest')
            Accs[k][i] = acc
            Scores[k][i] = score
            F1s[k][i] = f1
            Recalls[k][i] = recall
            #CC = np.mean(feature_indexs,0)
            #CC = np.int64(CC == 1.0)
            #index_features = np.arange(Feature_Matrix.shape[1])
            #CC = index_features*CC
            #index_CC = list(filter(lambda a:a>0, CC))
            #scores_result[k,i] = scores_mean
            print('Loop: %2d, feature: %2d' % (k + 1, num_features[i]))
    Accs_final = np.mean(Accs, axis = 0).reshape(1,-1)
    Scores_final = np.mean(Scores, axis = 0).reshape(1,-1)
    F1s_final = np.mean(F1s, axis = 0).reshape(1,-1)
    Recalls_final = np.mean(Recalls, axis = 0).reshape(1,-1)
    final_result = np.concatenate((Accs_final,Scores_final,F1s_final,Recalls_final), axis = 0)
    # file=open(scores_path,'w')
    # for m in range(len(scores_final)):
    #     file.writelines((str(scores_final[m])+"\n"));
    # file.close() 
    # result_root = 'F:/Documents/Desktop/SvmScript/SVM/python/final_result/all_nodule_new'
    acc_path = os.path.join(result_root, 'final.txt')
    file=open(acc_path,'w')
    for i in range(len(final_result)):
        file.writelines((str(final_result[i])+"\n"));
    file.close()
            
                
                
    # import math
    # FC = r'F:\Documents\Desktop\SvmScript\DATA\WUXI\wuxi_172_FC_MC_SC\oriVector\MN\KLS_GMV_withoutsub.mat'
    # KLS = r'F:\Documents\Desktop\SvmScript\DATA\WUXI\wuxi_172_FC_MC_SC\oriVector\JSD'
    # all_mudule = r'F:\Documents\Desktop\SvmScript\DATA\WUXI\wuxi_172_FC_MC_SC\oriVector\all_module'
    # label_path = r'F:\Documents\Desktop\SvmScript\DATA\WUXI\wuxi_172
    # _FC_MC_SC\label.mat'
    # Label = scio.loadmat(label_path)['label'].reshape(-1)
    # rf_params = {
    # 'C': [0.0001,0.001, 0.01,0.1,1,10,100,1000,10000],
    # "kernel":['linear','poly','rbf','sigmoid']
    # }
    # num_features = np.arange(50, 3001, 50)
    # Label = np.int64(Label>0)
    # scores_final = np.zeros((12,len(num_features)))
    # j = 0
    # for root, dirs, files in os.walk(all_mudule):
    #     for file in files:
    #         # FC DTI GMV
    #         path = os.path.join(root, file)
    #         feature = scio.loadmat(path)
    #         feature = feature[list(feature.keys())[-1]]
    #         #归一化
    #         feature_mean = np.mean(feature,1).reshape(-1,1)
    #         feature_std = np.std(feature,1).reshape(-1,1)
    #         feature_n = (feature - feature_mean)/feature_std
    #         if j == 0:
    #             feature_final = feature_n
    #         else:
    #             feature_final = np.concatenate((feature_final, feature_n), axis=1)
    #         j = j + 1
    # Feature_Matrix = feature_final;
    
    
    
    # # from sklearn.feature_selection import RFE
    # # import time
    # # X_Train, X_Test, Y_Train, Y_Test = train_test_split(Feature_Matrix, Label, test_size = 0.30,  random_state = 101) 
    
    # # model = svm.SVC(C = 1,kernel = 'linear', gamma='scale')
    # # rfe = RFE(model)
    # # start = time.process_time()
    # # RFE_X_Train = rfe.fit_transform(X_Train,Y_Train)
    # # RFE_X_Test = rfe.transform(X_Test)
    # # rfe = rfe.fit(RFE_X_Train,Y_Train)
    # # print(time.process_time() - start)
    # # print("Overall Accuracy using RFE: ", rfe.score(RFE_X_Test,Y_Test))
    # '''
    # 2 找到多模态分类最高对应的特征数量
    # '''
    # result_mean = np.zeros((5,len(num_features)))
    # # result_finale = np.mean(result_mean,axis = 0)
    # CC_results = np.zeros((5,90405))
    # for k in range(5):
    #     scores_result = np.zeros((12,len(num_features)))
    #     CC_result = np.zeros((90405,len(num_features)))
    #     for i in range(len(num_features)):
    #         #Result_root = os.path.join(result_root, ("feature_" + str(num_features[i])))
            
    #         scores_mean, scores_std, scores, feature_indexs = SVM_test(Feature_Matrix, Label, 10, rf_params, num_features[i])
    #         CC = np.mean(feature_indexs,0)
    #         CC = np.int64(CC == 1.0)
    #         index_features = np.arange(Feature_Matrix.shape[1])
    #         CC_index = index_features*CC
    #         index_CC = list(filter(lambda a:a>0, CC_index))
    #         scores.append(scores_mean)
    #         scores.append(scores_std)
    #         spd = np.array(scores)
    #         scores_result[:,i] = scores
    #         CC_result[:,i] = CC
    #     result_mean[k,:] = scores_result[10,:]
    #     CC_results[k,:] = np.sum(CC_result,axis = 1).T
        
    # result_name = 'all_mudule' + str(3000)
    # result_path = 'F:/Documents/Desktop/SvmScript/SVM/python/final_result/' + result_name
    # if not os.path.exists(result_path):
    #     os.makedirs(result_path)
    # scores_path = 'F:/Documents/Desktop/SvmScript/SVM/python/final_result/' + result_name+ '/scores_results.txt'
    # file=open(scores_path,'w')
    # for i in range(len(scores_result)):
    #     file.writelines((str(scores_result[i])+"\n"));
    # file.close() 
    # CC_path = 'F:/Documents/Desktop/SvmScript/SVM/python/final_result/' + result_name+ '/index_CC.txt'
    # np.savetxt(CC_path,CC_results,fmt='%d')
    
    # result_finale = np.mean(result_mean,axis = 0)
    
    # # 选择特征
    
    # choose_feature = np.array([result_finale,num_features])
    # choose_feature2 = choose_feature.T[np.lexsort(-choose_feature[::-1,:])].T
    # choose_number = choose_feature2[1][1].astype('int')
    # '''
    # 3 feature = 1950 三模态分类 存进result_all
    # 提取每一个模态被选出来的特征 进行单独分类 存进数组result_single
    
    # '''
    # scores_final = np.zeros((4,5))
    # for j in range(5):
        
    #     strtfdKFold = StratifiedKFold(n_splits=10, shuffle=True)
        
    #     kfold = strtfdKFold.split(Feature_Matrix, Label)
    #     scores_FC = []
    #     scores_SC = []
    #     scores_MC = []
    #     scores = []
        
    #     for k, (train, test) in enumerate(kfold):
    #         train_outer = np.array(Feature_Matrix)[train,:]
    #         test_outer = np.array(Feature_Matrix)[test,:]
    #         label_outer = Label[train]
    #         index = np.arange(len(label_outer))
    #         index_features = np.arange(train_outer.shape[1])
    #         ## 特征选择 双样本t检验
    #         index_NC = (index*label_outer).tolist()
    #         index_NC = list(filter(lambda a:a>0, index_NC))
    #         index_SZ = (index*(1-label_outer)).tolist()
    #         index_SZ = list(filter(lambda a:a>0, index_SZ))
    #         train_outerNC = train_outer[index_NC,:]
    #         train_outerSZ = train_outer[index_SZ,:]
    #         t, pval = stats.ttest_ind(train_outerNC, train_outerSZ) 
    #         #找出选取的特征
    #         tmp = np.array([pval,index_features])
    #         #x = tmp.T[np.lexsort(tmp[::-1,:])].T
    #         tmp = tmp.T[np.lexsort(tmp[::-1,:])].T
    #         turelabel = tmp[:][1].astype('int')
    #         feature_index = turelabel[:choose_number]
    #         FC_index = np.array([a for a in feature_index if a<30135])
    #         SC_index = np.array([a for a in feature_index if 30135<a and a<60270])
    #         MC_index = np.array([a for a in feature_index if 60270<a])
    #         # 特征选择阈值结果
    #         # p_label = np.int64(pval<0.1)
    #         #feature_index = index_features*p_label
    #         #feature_index = list(filter(lambda a:a>0, feature_index))
    #         ## single module
    #         FC_train = train_outer[:,FC_index]
    #         FC_test = test_outer[:,FC_index]
            
    #         SC_train = train_outer[:,SC_index]
    #         SC_test = test_outer[:,SC_index]
            
    #         MC_train = train_outer[:,MC_index]
    #         MC_test = test_outer[:,MC_index]
            
    #         model_val = svm.SVC(gamma='scale')
    #         grid_FC = GridSearchCV(model_val, rf_params, cv=10, scoring='accuracy')
    #         grid_FC.fit(FC_train, label_outer)
            
    #         PAR_FC = grid_FC.best_params_
    #         model_FC = svm.SVC(C = PAR_FC['C'],kernel = PAR_FC['kernel'],gamma='scale')
    #         model_FC.fit(FC_train, label_outer)
    #         score_FC = model_FC.score(FC_test, Label[test])
    #         scores_FC.append(score_FC)
    #         print('FC module Fold: %2d, Accuracy: %.3f' % (k+1, score_FC))
            
    #         model_val = svm.SVC(gamma='scale')
    #         grid_SC = GridSearchCV(model_val, rf_params, cv=10, scoring='accuracy')
    #         grid_SC.fit(SC_train, label_outer)
            
    #         PAR_SC = grid_FC.best_params_
    #         model_SC = svm.SVC(C = PAR_SC['C'],kernel = PAR_SC['kernel'],gamma='scale')
    #         model_SC.fit(SC_train, label_outer)
    #         score_SC = model_SC.score(SC_test, Label[test])
    #         scores_SC.append(score_SC)
    #         print('SC module Fold: %2d, Accuracy: %.3f' % (k+1, score_SC))
            
    #         model_val = svm.SVC(gamma='scale')
    #         grid_MC = GridSearchCV(model_val, rf_params, cv=10, scoring='accuracy')
    #         grid_MC.fit(MC_train, label_outer)
            
    #         PAR_MC = grid_FC.best_params_
    #         model_MC = svm.SVC(C = PAR_MC['C'],kernel = PAR_MC['kernel'],gamma='scale')
    #         model_MC.fit(MC_train, label_outer)
    #         score_MC = model_MC.score(MC_test, Label[test])
    #         scores_MC.append(score_MC)
    #         print('MC module Fold: %2d, Accuracy: %.3f' % (k+1, score_MC))
            
    #         ## all module
    #         train_selected = train_outer[:,feature_index]
    #         test_selected = test_outer[:,feature_index]
    #         #feature_indexs[k,feature_index] = 1
    #         ##网格寻优
    #         model_val = svm.SVC(gamma='scale')
    #         grid = GridSearchCV(model_val, rf_params, cv=10, scoring='accuracy')
    #         grid.fit(train_selected, label_outer)
    #         #print(grid.best_params_)
            
    #         PAR = grid.best_params_
    #         ##test
    #         model = svm.SVC(C = PAR['C'],kernel = PAR['kernel'],gamma='scale')
    #         model.fit(train_selected, label_outer)
    #         score = model.score(test_selected, Label[test])
    #         scores.append(score)
    #         print('All module Fold: %2d, Accuracy: %.3f' % (k+1, score))
    #     scores_final[:,j] = np.array([np.mean(scores),
    #                                  np.mean(scores_FC),
    #                                  np.mean(scores_SC),
    #                                  np.mean(scores_MC)])
    #     print('\n\nCross-Validation accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))
        
    # ACC_final = np.mean(scores_final,axis = 1)
    
    
    
    # scores_mean, scores_std, scores, feature_indexs = SVM_test(Feature_Matrix, Label, 10, rf_params, 1900)
    # CC = np.mean(feature_indexs,0)
    # CC = np.int64(CC == 1.0)
    # index_features = np.arange(Feature_Matrix.shape[1])
    # CC = index_features*CC
    # index_CC = list(filter(lambda a:a>0, CC))
    
    # scores.append(scores_mean)
    # scores.append(scores_std)
    # spd = np.array(scores)
    
    # np.savetxt(r'F:\Documents\Desktop\SvmScript\SVM\python\final_result\all_module\index_CC.txt',index_CC,fmt='%d')
    # result_name = 'JSD'
    # scores_path = 'F:/Documents/Desktop/SvmScript/SVM/python/final_result/' + result_name+ '/scores_results.txt'
    # file=open(scores_path,'w')
    # for i in range(len(scores_result)):
    #     file.writelines((str(scores_result[i])+"\n"));
    # file.close() 
    #X_train, X_val, y_train, y_val = train_test_split(Feature_Matrix, Label, test_size=0.25)
    
    
    # Feature_Matrix_SZ = Gradient_G2_SZ_Matrix_trans
    # Feature_Matrix_NC = Gradient_G2_NC_Matrix_trans
    
    # Feature_Matrix_SZ = DET_SZ_Matrix
    # Feature_Matrix_NC = DET_NC_Matrix
    #
    # Feature_Matrix_SZ = ENTRY_SZ_Matrix
    # Feature_Matrix_NC = ENTRY_NC_Matrix
    #
    # Feature_Matrix_SZ = L_SZ_Matrix
    # Feature_Matrix_NC = L_NC_Matrix
    
    # Feature_Matrix_SZ = time_sz_Matrix
    # Feature_Matrix_NC = time_hc_Matrix
    
    
    # model = XGBC(n_estimators=120, random_state=1, use_label_encoder=False, max_depth=4, learning_rate=0.1, colsample_bytree=0.5)
    # model = GTBC(n_estimators=120, learning_rate=0.1, max_depth=5, min_samples_split=60, min_samples_leaf=10)
    #model = XGBC(n_estimators=120, random_state=1, use_label_encoder=False, max_depth=4, learning_rate=0.1, colsample_bytree=0.5)
    #clf = SelectFromModel(model, threshold=0.0052).fit_transform(Feature_Matrix, Label) 
    
    
    #importances = clf.feature_importances_
    #indices = np.argsort(importances)[::-1]
    
    
    
    
    
    # X_train, X_val, y_train, y_val = train_test_split(Feature_Matrix, Label, test_size=0.25)
    
    # sel_fea = RFE(estimator=model, n_features_to_select=100).fit_transform(X_val, y_val)
    
    # # importances_ = model.fit(Feature_Matrix, Label).feature_importances_
    # # importances = model.feature_importances_.tolist()
    # importances_ = model.fit(X_val, y_val).coef_
    # importances = model.coef_[0]
    
    # # indices = np.argsort(importances)[::-1].tolist()
    # indices = np.argsort(abs(importances))[::-1].tolist()
    # num_features = 50
    # Selected_feature = {}
    # Selected_feature_index = indices[:num_features]
    # for j in range(num_features):
    #     Selected_feature[j] = importances[indices[j]]
    # Selected_feature = pd.DataFrame(Selected_feature.items()).transpose()
    # Selected_feature.columns = Selected_feature_index
    # Selected_feature = Selected_feature.iloc[1, :]
    
    # # plt.figure()
    # # plt.title("Feature importances")
    # # plt.bar(range(num_features), abs(Selected_feature), color="g", align="center")
    # # plt.xticks(range(num_features), Selected_feature_index, rotation = 90)
    # # plt.xlim([-1, num_features])
    # # plt.show()
    
    # X_train= X_train.loc[:, Selected_feature_index]
    # new_index = {}
    # for i in range(num_features):
    #     new_index[i] = i
    # X_train.columns = new_index
    # X_train = X_train.values
    
    # X_val = X_val.loc[:, Selected_feature_index]
    # X_val.columns = new_index
    # X_val = X_val.values
    
    # strtfdKFold = StratifiedKFold(n_splits=5, shuffle=True)
    
    # kfold = strtfdKFold.split(X_train, y_train)
    # scores = []
    
    # for k, (train, test) in enumerate(kfold):
    #     model.fit(X_train[train, :], y_train[train])
    #     score = model.score(X_train[test, :], y_train[test])
    #     scores.append(score)
    #     print('Fold: %2d, Accuracy: %.3f' % (k + 1, score))
    # print('\n\nCross-Validation accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    
    
    # kfold2 = strtfdKFold.split(X_val, y_val)
    # scores2 = []
    # for k, (train, test) in enumerate(kfold2):
    #     model.fit(X_val[train, :],  y_val[train])
    #     score = model.score(X_val[test, :],  y_val[test])
    #     scores2.append(score)
    #     print('Fold: %2d, Accuracy: %.3f' % (k + 1, score))
    # print('\n\nCross-Validation accuracy: %.3f +/- %.3f' % (np.mean(scores2), np.std(scores2)))
    
    
