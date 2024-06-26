'''
数据来源：有毒性和无毒性数据都来自Ms_Hou，数据集中共包含191个数据
6种分子表征方法：Avalon、ECFP4、RDkit、Maccs、Brics
三类建模方法:(1)SVM; (2)Bagging:RF  (3)Boosting:GBDT
优化模型超参数：TPE
'''

from utils_fingerprint import *
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef
import os
import xlwt
import joblib

from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
num_data = 10

'''6种分子表征形式：不同分子描述符'''
for res in range(0, 4):
    repre_list = ['Avalon', 'ECFP', 'FCFP', 'Rdkit', 'MACCS',
                  'Chemphy', 'LSTM', 'GRU', 'MPNN']

    s = repre_list[res] + '_RFE'
    load_data_path = os.path.join('./data/save_data_random_FS/', s)

    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet('Exp_results')
    sheet.write(0, 1, 'Train_ACC')
    sheet.write(0, 2, 'Test_ACC')
    sheet.write(0, 3, 'Train_BA')
    sheet.write(0, 4, 'Test_BA')
    sheet.write(0, 5, 'Train_F1')
    sheet.write(0, 6, 'Test_F1')
    sheet.write(0, 7, 'Train_MCC')
    sheet.write(0, 8, 'Test_MCC')
    sheet.write(0, 9, 'Feature_dim')

    train_accuracy_all = 0
    test_accuracy_all = 0

    train_BA_all = 0
    test_BA_all = 0

    train_f1_score_all = 0
    test_f1_score_all = 0

    train_mcc_all = 0
    test_mcc_all = 0
    r = 0

    for i in range(0, num_data):
        train_x = np.load(os.path.join(load_data_path, s + '_' + 'train_x%d.npy' % i))
        print(os.path.join(load_data_path, s + '_' + 'train_x%d.npy' % i))
        train_y = np.load(os.path.join(load_data_path, s + '_' + 'train_y%d.npy' % i))
        test_x = np.load(os.path.join(load_data_path, s + '_' + 'test_x%d.npy' % i))
        test_y = np.load(os.path.join(load_data_path, s + '_' + 'test_y%d.npy' % i))
        feature_dim = len(train_x[1])

        if s == 'Chemphy':
            train_x = min_max_scaler.fit_transform(train_x)
            test_x = min_max_scaler.fit_transform(test_x)

        '''不同建模方法：RF、SVM、GBDT'''
        # '''RF_Classifier_model'''
        #
        # save_model_path = os.path.join('./data/FS_Random_RF_Results/', s)
        # os.makedirs(save_model_path, exist_ok=True)
        #
        # save_data_path = './data/FS_Random_RF_Results/'
        # os.makedirs(save_data_path, exist_ok=True)
        #
        # from sklearn.ensemble import RandomForestClassifier
        # from hyperopt import fmin, tpe, hp, partial
        #
        #
        # def RF(argsDict):
        #     max_depth = argsDict["max_depth"] + 5  # [5, 20]
        #     n_estimators = argsDict['n_estimators'] * 5 + 50  # [50, 100]
        #     max_features = argsDict['max_features'] * 5 + 5  # [5, 30]
        #     global train_x, train_y
        #     RF_model = RandomForestClassifier(n_estimators=n_estimators,
        #                                       max_features=max_features,
        #                                       max_depth=max_depth,
        #                                       random_state=0)
        #
        #     RF_model.fit(train_x, train_y)
        #     pre_train_y = RF_model.predict(train_x)
        #     pre_test_y = RF_model.predict(test_x)
        #     train_accuracy = accuracy_score(train_y, pre_train_y)
        #     test_accuracy = accuracy_score(test_y, pre_test_y)
        #     ave_accuracy = (train_accuracy + test_accuracy) / 2
        #     return -ave_accuracy
        #
        #
        # space = {"max_depth": hp.randint("max_depth", 15),
        #          "n_estimators": hp.randint("n_estimators", 10),  # [0,1,2,3,4,5] -> [50,100]
        #          "max_features": hp.randint("max_features", 6)}  # [0,1,2,3,4,5,6] -> [5,35]}
        #
        # algo = partial(tpe.suggest, n_startup_jobs=1)
        # best = fmin(RF, space, algo=algo, max_evals=50)
        # print(best)
        # print(RF(best))
        #
        # max_depth = best["max_depth"] + 5  # [5, 20]
        # n_estimators = best['n_estimators'] * 5 + 50  # [50, 100]
        # max_features = best['max_features'] * 5 + 5  # [5, 30]
        #
        # print("max_depth:" + str(max_depth))
        # print("n_estimator:" + str(n_estimators))
        # print("max_features:" + str(max_features))
        #
        # RF_model = RandomForestClassifier(n_estimators=n_estimators,
        #                                   max_features=max_features,
        #                                   max_depth=max_depth,
        #                                   random_state=0)
        #
        # RF_model.fit(train_x, train_y)
        # pre_train_y = RF_model.predict(train_x)
        # pre_test_y = RF_model.predict(test_x)
        # joblib.dump(RF_model, os.path.join(save_model_path, 'FS_BM_RF_' + s + str(i) + '.pkl'))

        # '''SVM_classifier_Model'''
        # save_data_path = './data/FS_Random_SVM_Results/'
        # os.makedirs(save_data_path, exist_ok=True)
        #
        # save_model_path = os.path.join('./data/FS_Random_SVM_Results/', s)
        # os.makedirs(save_model_path, exist_ok=True)
        #
        # from sklearn.svm import SVC
        # from hyperopt import fmin, tpe, hp, partial
        #
        #
        # def SVM(argsDict):
        #     C = argsDict['C']
        #     shrinking = argsDict['shrinking']
        #     kernel = argsDict['kernel']
        #     gamma = argsDict['gamma']
        #     global train_x, train_y
        #
        #     SVM_Model = SVC(random_state=0,
        #                     C=C,
        #                     shrinking=shrinking,
        #                     gamma=gamma,
        #                     kernel=kernel,
        #                     probability=True)
        #
        #     SVM_Model.fit(train_x, train_y)
        #     pre_train_y = SVM_Model.predict(train_x)
        #     pre_test_y = SVM_Model.predict(test_x)
        #     train_accuracy = accuracy_score(train_y, pre_train_y)
        #     test_accuracy = accuracy_score(test_y, pre_test_y)
        #     ave_accuracy = (train_accuracy + test_accuracy) / 2
        #     return -ave_accuracy
        #
        #
        # space = {'C': hp.uniform('C', 0.001, 1000),
        #          'shrinking': hp.choice('shrinking', [True, False]),
        #          'kernel': hp.choice('kernel', ['rbf', 'sigmoid', 'poly']),
        #          'gamma': hp.choice('gamma', ['auto', hp.uniform('gamma_function', 0.0001, 8)])}
        #
        # algo = partial(tpe.suggest, n_startup_jobs=1)
        # best = fmin(SVM, space, algo=algo, max_evals=50)
        # print(best)
        #
        # gamma_para = ['auto', hp.uniform('gamma_function', 0.0001, 8)]
        # kernel_para = ['rbf', 'sigmoid', 'poly']
        # C = best['C']
        # if best['gamma'] == 0:
        #     gamma = gamma_para[best['gamma']]
        # else:
        #     gamma = best['gamma_function']
        # kernel = kernel_para[best['kernel']]
        # shrinking = best['shrinking']
        #
        # SVM_Model = SVC(random_state=0,
        #                 C=C,
        #                 shrinking=shrinking,
        #                 gamma=gamma,
        #                 kernel=kernel)
        #
        # SVM_Model.fit(train_x, train_y)
        # pre_train_y = SVM_Model.predict(train_x)
        # pre_test_y = SVM_Model.predict(test_x)
        # joblib.dump(SVM_Model, os.path.join(save_model_path, 'FS_BM_SVM_' + s + str(i) + '.pkl'))

        # '''GBDT model'''
        save_data_path = './data/FS_Random_GBDT_Results/'
        os.makedirs(save_data_path, exist_ok=True)

        save_model_path = os.path.join('./data/FS_Random_GBDT_Results/', s)
        os.makedirs(save_model_path, exist_ok=True)

        from sklearn.ensemble import GradientBoostingClassifier
        from hyperopt import fmin, tpe, hp, partial, rand


        def GBDT(argsDict):
            max_depth = argsDict["max_depth"] + 5
            n_estimators = argsDict['n_estimators'] * 5 + 30
            learning_rate = argsDict["learning_rate"] * 0.02 + 0.05
            subsample = argsDict["subsample"] * 0.1 + 0.7
            global train_x, train_y

            GBDT_model = GradientBoostingClassifier(random_state=0,
                                                    max_depth=max_depth,  # 最大深度
                                                    n_estimators=n_estimators,  # 树的数量
                                                    learning_rate=learning_rate,  # 学习率
                                                    subsample=subsample)  # 采样数

            GBDT_model.fit(train_x, train_y)
            pre_train_y = GBDT_model.predict(train_x)
            pre_test_y = GBDT_model.predict(test_x)
            train_accuracy = accuracy_score(train_y, pre_train_y)
            test_accuracy = accuracy_score(test_y, pre_test_y)
            ave_accuracy = (train_accuracy + test_accuracy) / 2
            # metric = cross_val_score(model, train_x, train_y, cv=5, scoring="roc_auc").mean()
            # print(train_accuracy, test_accuracy)
            return -ave_accuracy


        space = {"max_depth": hp.randint("max_depth", 5),  # [0-5]-->[5, 10]
                 "n_estimators": hp.randint("n_estimators", 10),  # [0-10] -> [30,80]
                 "learning_rate": hp.randint("learning_rate", 5),  # [0-5] -> [0.05,0.15]
                 "subsample": hp.randint("subsample", 3)}  # [0-3] -> [0.7-1.0]

        algo = partial(tpe.suggest, n_startup_jobs=1)
        best = fmin(GBDT, space, algo=algo, max_evals=50)
        print(best)
        print(GBDT(best))

        max_depth = best["max_depth"] + 5
        n_estimators = best['n_estimators'] * 5 + 30
        learning_rate = best["learning_rate"] * 0.02 + 0.05
        subsample = best["subsample"] * 0.1 + 0.7
        print("max_depth:" + str(max_depth))
        print("n_estimator:" + str(n_estimators))
        print("learning_rate:" + str(learning_rate))
        print("subsample:" + str(subsample))

        GBDT_model = GradientBoostingClassifier(random_state=0,
                                                max_depth=max_depth,  # 最大深度
                                                n_estimators=n_estimators,  # 树的数量
                                                learning_rate=learning_rate,  # 学习率
                                                subsample=subsample, )  # 采样数

        GBDT_model.fit(train_x, train_y)
        pre_train_y = GBDT_model.predict(train_x)
        pre_test_y = GBDT_model.predict(test_x)
        joblib.dump(GBDT_model, os.path.join(save_model_path, 'FS_BM_GBDT_' + s + str(i) + '.pkl'))

        train_accuracy = accuracy_score(train_y, pre_train_y)
        test_accuracy = accuracy_score(test_y, pre_test_y)

        train_BA = balanced_accuracy_score(train_y, pre_train_y)
        test_BA = balanced_accuracy_score(test_y, pre_test_y)

        train_f1_score = f1_score(train_y, np.array(pre_train_y))
        test_f1_score = f1_score(test_y, np.array(pre_test_y))

        train_mcc = matthews_corrcoef(train_y, np.array(pre_train_y))
        test_mcc = matthews_corrcoef(test_y, np.array(pre_test_y))

        sheet.write(r + 1, 0, 'Model' + str(i))
        sheet.write(r + 1, 1, '%.2f%%' % (train_accuracy * 100))
        sheet.write(r + 1, 2, '%.2f%%' % (test_accuracy * 100))
        sheet.write(r + 1, 3, '%.2f%%' % (train_BA * 100))
        sheet.write(r + 1, 4, '%.2f%%' % (test_BA * 100))
        sheet.write(r + 1, 5, '%0.4f' % (train_f1_score))
        sheet.write(r + 1, 6, '%0.4f' % (test_f1_score))
        sheet.write(r + 1, 7, '%0.4f' % (train_mcc))
        sheet.write(r + 1, 8, '%0.4f' % (test_mcc))
        sheet.write(r + 1, 9, '%d' % (feature_dim))

        train_accuracy_all += train_accuracy
        test_accuracy_all += test_accuracy

        train_BA_all += train_BA
        test_BA_all += test_BA

        train_f1_score_all += train_f1_score
        test_f1_score_all += test_f1_score

        train_mcc_all += train_mcc
        test_mcc_all += test_mcc

        if r == num_data - 1:
            sheet.write(r + 2, 0, 'average_all')
            sheet.write(r + 2, 1, '%.2f%%' % (train_accuracy_all / num_data * 100))
            sheet.write(r + 2, 2, '%.2f%%' % (test_accuracy_all / num_data * 100))
            sheet.write(r + 2, 3, '%.2f%%' % (train_BA_all / num_data * 100))
            sheet.write(r + 2, 4, '%.2f%%' % (test_BA_all / num_data * 100))

            sheet.write(r + 2, 5, '%0.4f' % (train_f1_score_all / num_data))
            sheet.write(r + 2, 6, '%0.4f' % (test_f1_score_all / num_data))
            sheet.write(r + 2, 7, '%0.4f' % (train_mcc_all / num_data))
            sheet.write(r + 2, 8, '%0.4f' % (test_mcc_all / num_data))
            sheet.write(r + 2, 9, '%d' % (feature_dim))
            break
        r = r + 1
    workbook.save(os.path.join(save_data_path, 'Results_%s.xls' % (s)))
