import csv
import random
from operator import le
import os
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from baggingPU import BaggingClassifierPU
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
import logging
import json
import rocksdb
import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMRegressor
from sklearn.svm import OneClassSVM
from BaseSVDD import BaseSVDD
# from deepSVDD import DeepSVDD
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RansomwareAddressDetection")


def load_labeled_address_data(tag):
    train_data_list = []
    label_data_list = []
    test_data_list = []
    label_test_list = []
    ransomware_address_path = '../nfs/zhaoyu/ransom_addr/'
    tag_list = os.listdir(ransomware_address_path)
    for tag_file in tag_list:
        if tag_file == tag + '_addr_feature.csv':
            continue
        tag_path = ransomware_address_path + tag_file
        ransomware_file = open(tag_path, "r")
        reader = csv.reader(ransomware_file)
        for row in reader:
            address = row[0]
            features = row[1:]
            for i in range(len(features)):
                if features[i] == 'inf' or features[i] == 'nan':
                    features[i] = 0
            train_data_list.append(features)
            label_data_list.append(1)

    tag_path = ransomware_address_path + tag + '_addr_feature.csv'
    ransomware_file = open(tag_path, "r")
    reader = csv.reader(ransomware_file)
    for row in reader:
        address = row[0]
        features = row[1:]
        for i in range(len(features)):
            if features[i] == 'inf' or features[i] == 'nan':
                features[i] = 0
        test_data_list.append(features)
        label_test_list.append(1)
    return train_data_list, label_data_list, test_data_list, label_test_list


def load_all_period_year_data(first_date, last_date):
    unlabeled_feature_list = []
    unlabeled_label_list = []
    unlabel_test_list = []
    unlabel_label_list = []

    #   这里选取白样本地址
    all_white_addr_info = load_all_white_addr()
    target_white_addr = set()
    train_white_addr = set()
    for one_addr in all_white_addr_info:
        if (all_white_addr_info[one_addr][1] < first_date) or (all_white_addr_info[one_addr][0] > last_date):
            train_white_addr.add(one_addr)
        if (all_white_addr_info[one_addr][0] > first_date) and (all_white_addr_info[one_addr][1] < last_date):
            target_white_addr.add(one_addr)
    if len(target_white_addr) < 10000:
        for one_addr in all_white_addr_info:
            if (all_white_addr_info[one_addr][0] > first_date) or (all_white_addr_info[one_addr][1] < last_date):
                target_white_addr.add(one_addr)
    else:
        target_white_addr = list(target_white_addr)[0:10000]

    while len(target_white_addr) < 10000:
        for one_addr in all_white_addr_info:
            target_white_addr.add(one_addr)
    target_white_addr = set(target_white_addr)
    #   地址选取完毕
    # ../cd_data/cascade_feature_path
    white_addr_feature_path = '../nfs/wk/TIFS/cascade_feature_path/random_addr/combined_random_cascade_feature.csv'
    sample_file = open(white_addr_feature_path, "r")
    reader = csv.reader(sample_file)
    for row in reader:
        address = row[0]
        if address in train_white_addr:
            features = row[1:]
            unlabeled_feature_list.append(features)
            unlabeled_label_list.append(0)
            for i in range(len(features)):
                if features[i] == 'inf' or features[i] == 'nan':
                    features[i] = 0
        elif address in target_white_addr:
            features = row[1:]
            unlabel_test_list.append(features)
            unlabel_label_list.append(0)
            for i in range(len(features)):
                if features[i] == 'inf' or features[i] == 'nan':
                    features[i] = 0

    return unlabeled_feature_list, unlabeled_label_list, unlabel_test_list, unlabel_label_list


def load_rs_data_test(rs, first_date, last_date):
    train_data_list = []
    label_data_list = []
    unlabeled_feature_list = []
    unlabeled_label_list = []
    # cryptolocker 2013
    # cryptowall 2014
    # cryptxxx 2016
    # locky 2016
    # cerber 2016
    # wannacry 2017
    # dmalocker 2016
    # rs_family = {'cryptolocker':2013,'cryptowall':2014,'cryptxxx':2016,'locky':2016,'cerber':2016, 'wannacry':2017
    #              ,'dmalocker':2016}

    rs_family = {'CryptoLocker_address_cascade_features': 2013, 'CryptoWall_address_cascade_features': 2014,
                 'CryptXXX_address_cascade_features': 2016, 'Locky_address_cascade_features': 2016,
                 'Cerber_address_cascade_features': 2016}
    rs_feature_path = '../nfs/zhaoyu/ransom_addr/' + rs + '_addr_feature.csv'
    ransomware_file = open(rs_feature_path, "r")
    reader = csv.reader(ransomware_file)
    for row in reader:
        address = row[0]
        features = row[1:]
        train_data_list.append(features)
        label_data_list.append(1)
        for i in range(len(features)):
            if features[i] == 'inf' or features[i] == 'nan':
                features[i] = 0

    #   这里选取白样本地址
    all_white_addr_info = load_all_white_addr()
    target_white_addr = set()
    for one_addr in all_white_addr_info:
        if (all_white_addr_info[one_addr][0] > first_date) and (all_white_addr_info[one_addr][1] < last_date):
            target_white_addr.add(one_addr)
    if len(target_white_addr) < 10000:
        for one_addr in all_white_addr_info:
            if (all_white_addr_info[one_addr][0] > first_date) or (all_white_addr_info[one_addr][1] < last_date):
                target_white_addr.add(one_addr)
    else:
        target_white_addr = list(target_white_addr)[0:10000]

    while len(target_white_addr) < 10000:
        for one_addr in all_white_addr_info:
            target_white_addr.add(one_addr)
    target_white_addr = set(target_white_addr)
    #   地址选取完毕

    white_addr_feature_path = '../nfs/wk/TIFS/cascade_feature_path/random_addr/combined_random_cascade_feature.csv'
    sample_file = open(white_addr_feature_path, "r")
    reader = csv.reader(sample_file)
    for row in reader:
        address = row[0]
        if address in target_white_addr:
            features = row[1:]
            unlabeled_feature_list.append(features)
            unlabeled_label_list.append(0)
            # unlabeled_label_list.append(-1)
            for i in range(len(features)):
                if features[i] == 'inf' or features[i] == 'nan':
                    features[i] = 0

    return train_data_list, label_data_list, unlabeled_feature_list, unlabeled_label_list


def load_unknown_data(tag_name):
    rs_family = {'CryptoLocker': '2013-09-08', 'Zeppelin': '2019-12-7', 'CryptoWall': '2014-05-01', 'Locky': '2016-02-16',
                 'Cerber': '2016-04-09', 'CryptXXX': '2016-04-26', 'DMALockerv3': '2016-01-01', 'SamSam': '2016-03-31',
                 'DMALocker': '2016-01-01', 'NoobCrypt': '2016-09-01', 'Makop': '2020-02-21', 'WannaCry': '2017-05-12',
                 'ryuk': '2018-08-21', 'CryptoDefense': '2014-03-20', 'lockbit': '2020-01-20', 'APT': '2015-02-25',
                 'sad': ' 2020-06-03', 'Globe': '2016-10-03', 'Sam': '2016-09-27', 'GlobeImposter': '2017-06-08',
                 'Globev3': '2017-02-01', 'CryptoTorLocker2015': '2015-01-01', 'XLockerv5.0': ' 2017-01-24',
                 'NotPetya': '2017-06-27', 'good': '2013-05-20', 'XTPLocker': ' 2016-05-09', 'Razy': '2019-01-01',
                 'Flyper': '2017-01-08', 'PopCornTime': '2014-02-28', 'VenusLocker': '2017-02-27', 'EDA2': '2016-03-01',
                 'CryptConsole': '2017-01-01', 'KeRanger': '2016-03-06', 'Xorist': '2016-11-13', 'hellokitty': '2021-02-13',
                 'Jigsaw': '2016-04-12', 'Kali': '2016-01-15', 'Avaddon': '2020-06-08', 'XLocker': '2014-01-08',
                 'mmm': '2018-01-01', 'ComradeCircle': '2016-10-01', 'iq': '2020-08-01', 'hydra': '2020-02-01',
                 'Cryptohitman': '2016-05-13', 'BadRabbit': '2017-10-24', 'NullByte': '2016-08-01', 'Exotic': '2017-01-06',
                 'Bucbi': '2012-06-02', 'GoldenEye': '2016-12-06', 'Phoenix': '2015-08-20', 'Miss': '2016-03-08',
                 'CryptoHost': '2015-08-03', 'ZCryptor': '2016-05-30', '7ev3n': '2016-01-28', 'CTB-Locker': '2014-07-27',
                 'Chimera': '2015-09-23', 'KillDisk': '2017-01-12', 'DoubleLocker': '2017-10-14',
                 'TeslaCrypt': '2015-03-06'}
    before_list = []
    rs_start_date = rs_family[tag_name]
    for one_rs in rs_family:
        if rs_family[one_rs] < rs_start_date:
            before_list.append(one_rs)

    train_data_list = []
    label_data_list = []
    test_data_list = []
    test_data_label_list = []

    ransomware_address_path = '../nfs/zhaoyu/ransom_addr/'
    for tag_file in before_list:
        tag_path = ransomware_address_path + tag_file + '_addr_feature.csv'
        ransomware_file = open(tag_path, "r")
        reader = csv.reader(ransomware_file)
        for row in reader:
            address = row[0]
            features = row[1:]
            for i in range(len(features)):
                if features[i] == 'inf' or features[i] == 'nan':
                    features[i] = 0
            train_data_list.append(features)
            label_data_list.append(1)

    tag_path = ransomware_address_path + tag_name + '_addr_feature.csv'
    ransomware_file = open(tag_path, "r")
    reader = csv.reader(ransomware_file)
    for row in reader:
        address = row[0]
        features = row[1:]
        for i in range(len(features)):
            if features[i] == 'inf' or features[i] == 'nan':
                features[i] = 0
        test_data_list.append(features)
        test_data_label_list.append(1)
        # 之前的rs地址特征，1，目标地址特征，1
    return train_data_list, label_data_list, test_data_list, test_data_label_list


def load_unknown_white_data(first_date, last_date):
    train_white_feature_list = []
    train_white_label_list = []
    white_feture_list = []
    white_label_list = []

    #   这里选取白样本地址
    all_white_addr_info = load_all_white_addr()
    target_white_addr = set()
    train_white_addr = set()
    for one_addr in all_white_addr_info:
        if (all_white_addr_info[one_addr][0] > first_date) and (all_white_addr_info[one_addr][1] < last_date):
            target_white_addr.add(one_addr)
        if (all_white_addr_info[one_addr][1] < first_date):
            train_white_addr.add(one_addr)
    if len(target_white_addr) < 40000:
        for one_addr in all_white_addr_info:
            if (all_white_addr_info[one_addr][0] > first_date) or (all_white_addr_info[one_addr][1] < last_date):
                target_white_addr.add(one_addr)
    else:
        target_white_addr = list(target_white_addr)[0:40000]

    while len(target_white_addr) < 40000:
        for one_addr in all_white_addr_info:
            target_white_addr.add(one_addr)
    target_white_addr = set(target_white_addr)
    
    #   地址选取完毕

    white_addr_feature_path = '../nfs/wk/TIFS/cascade_feature_path/random_addr/combined_random_cascade_feature.csv'
    sample_file = open(white_addr_feature_path, "r")
    reader = csv.reader(sample_file)
    for row in reader:
        address = row[0]
        if address in target_white_addr:
            features = row[1:]
            white_feture_list.append(features)
            white_label_list.append(0)
            for i in range(len(features)):
                if features[i] == 'inf' or features[i] == 'nan':
                    features[i] = 0
        elif address in train_white_addr:
            features = row[1:]
            train_white_feature_list.append(features)
            train_white_label_list.append(0)
            for i in range(len(features)):
                if features[i] == 'inf' or features[i] == 'nan':
                    features[i] = 0
    return train_white_feature_list, train_white_label_list, white_feture_list, white_label_list


def get_one_rs_active_date(rs):
    date_list = []
    for line in open('../cd_data/addr_active_time/' + rs + '_active.csv', 'r'):
        if '\n' in line:
            line = line[0:-1]
        line = line.split(',')
        start_date = line[1]
        end_date = line[2]
        date_list.append(start_date)
        date_list.append(end_date)

    rs_family_start_date = {'CryptoLocker': '2013-9-8', 'Zeppelin': '2019-12-7', 'CryptoWall': '2014-5-1',
                            'Locky': '2016-2-16',
                            'Cerber': '2016-4-9', 'CryptXXX': '2016-4-26', 'DMALockerv3': '2016-1-1',
                            'SamSam': '2016-3-31',
                            'DMALocker': '2016-1-1', 'NoobCrypt': '2016-9-1', 'Makop': '2020-2-21',
                            'WannaCry': '2017-5-12',
                            'ryuk': '2018-8-21', 'CryptoDefense': '2014-3-20', 'lockbit': '2020-1-20',
                            'APT': '2015-2-25',
                            'sad': ' 2020-6-3', 'Globe': '2016-10-3', 'Sam': '2016-9-27', 'GlobeImposter': '2017-6-8',
                            'Globev3': '2017-2-1', 'CryptoTorLocker2015': '2015-1-1', 'XLockerv5.0': ' 2017-1-24',
                            'NotPetya': '2017-6-27', 'good': '2013-5-20', 'XTPLocker': ' 2016-5-09', 'Razy': '2019-1-1',
                            'Flyper': '2017-1-8', 'PopCornTime': '2014-2-28', 'VenusLocker': '2017-2-27',
                            'EDA2': '2016-3-1',
                            'CryptConsole': '2017-1-1', 'KeRanger': '2016-3-6', 'Xorist': '2016-11-13',
                            'hellokitty': '2021-2-13',
                            'Jigsaw': '2016-4-12', 'Kali': '2016-1-15', 'Avaddon': '2020-6-8', 'XLocker': '2014-1-8',
                            'mmm': '2018-1-1', 'ComradeCircle': '2016-10-1', 'iq': '2020-8-1', 'hydra': '2020-2-1',
                            'Cryptohitman': '2016-5-13', 'BadRabbit': '2017-10-24', 'NullByte': '2016-8-1',
                            'Exotic': '2017-1-6',
                            'Bucbi': '2012-6-2', 'GoldenEye': '2016-12-6', 'Phoenix': '2015-8-20', 'Miss': '2016-3-8',
                            'CryptoHost': '2015-8-3', 'ZCryptor': '2016-5-30', '7ev3n': '2016-1-28',
                            'CTB-Locker': '2014-7-27',
                            'Chimera': '2015-9-23', 'KillDisk': '2017-1-12', 'DoubleLocker': '2017-10-14',
                            'TeslaCrypt': '2015-3-6'}
    last_date = max(date_list)
    return rs_family_start_date[rs], last_date


def load_all_white_addr():
    white_addr_time_loca = '../cd_data/white_addr_first_last_date.csv'
    addr_date_dict = {}
    for line in open(white_addr_time_loca, 'r'):
        if '\n' in line:
            line = line[0:-1]
        line = line.split(',')
        addr = line[0]
        start_date = line[1]
        end_date = line[2]
        addr_date_dict[addr] = [start_date, end_date]
    return addr_date_dict


def pu_learning_detection(rs):
    logger.info("loading dataset......")
    this_rs_first, this_rs_last = get_one_rs_active_date(rs)
    logger.info(this_rs_first)
    logger.info(this_rs_last)
    labeled_feature_list, labeled_list, unlabeled_feature_list, unlabeled_list = load_rs_data_test(rs, this_rs_first,
                                                                                                   this_rs_last)
    logger.info("labeled_num: " + str(len(labeled_feature_list)))
    logger.info("unlabeled_num: " + str(len(unlabeled_feature_list)))

    logger.info("loading dataset finished")

    labeled_feature_list.extend(unlabeled_feature_list)
    labeled_list.extend(unlabeled_list)
    labeled_feature_list = np.array(labeled_feature_list)
    labeled_list = np.array(labeled_list)

    train_x, test_x, train_y, test_y = train_test_split(labeled_feature_list, labeled_list, test_size=0.2)

    test_feature_list = np.array(test_x)
    test_feature_list = test_feature_list.astype(np.float64)

    test_label_list = np.array(test_y)
    logger.info('sum trainy ' + str(sum(train_y)))
    # bc = BaggingClassifierPU(
    #     GradientBoostingClassifier(),
    #     n_estimators=1000,  # 1000 trees as usual
    #     max_samples=sum(train_y),  # Balance the positives and unlabeled in each bag
    #     n_jobs=40  # Use all cores
    # )
    # bc.fit(train_x, train_y)

    #predict_results = bc.predict(test_feature_list)

    # lr = LogisticRegression()
    # lr.fit(train_x, train_y)
    # predict_results = lr.predict(test_feature_list)

    # svm_model = LinearSVC(penalty='l2', C=1.0)
    # svm_model.fit(train_x, train_y)
    # predict_results = svm_model.predict(test_feature_list)

    svm_model = SVC(C=1.0, kernel='poly')
    svm_model.fit(train_x, train_y)
    predict_results = svm_model.predict(test_feature_list)

    # xgb = XGBClassifier()
    # xgb.fit(train_x, train_y)
    # predict_results = xgb.predict(test_feature_list)

    # o_svm = OneClassSVM(kernel='linear', nu=0.1)
    # o_svm.fit(train_x, train_y)
    # predict_results = o_svm.predict(test_feature_list)

    # deep_SVDD = DeepSVDD(objective='one-class', nu=0.5)
    # deep_SVDD.set_network('mnist_LeNet')
    # train_x = train_x.astype(np.float64)
    # train_x = torch.from_numpy(train_x)
    #
    # deep_SVDD.train(train_x, device='cpu')

    # svdd = BaseSVDD(C=0.9, gamma=0.3, kernel='rbf', display='on', n_jobs=40)
    # train_x = np.array(train_x)
    # tmp = train_x.shape[0]
    # train_y_tmp = train_y.reshape(tmp, 1)
    # train_x = train_x.astype(np.float64)
    # svdd.fit(X=train_x, y=train_y_tmp)
    #
    # tmp2 = test_feature_list.shape[0]
    # test_label_list = test_label_list.reshape(tmp2, 1)
    # predict_results = svdd.predict(test_feature_list)

    # lgb = LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.05, n_estimators=20)
    # lgb.fit(train_x, train_y, eval_set=[(test_feature_list, test_label_list)], eval_metric='l1', early_stopping_rounds=5)
    # predict_results = lgb.predict(test_feature_list, num_iteration=lgb.best_iteration_)
    # print(roc_auc_score(test_label_list, predict_results))
    bitcoin_heist_test(predict_results, test_label_list)


def unknown_detection(rs):
    logger.info("loading dataset......")
    labeled_features_list, labeled_list, test_rs, test_rs_label = load_unknown_data(rs)
    this_rs_first, this_rs_last = get_one_rs_active_date(rs)
    unlabeled_features_list, unlabeled_list, test_white, test_white_label = load_unknown_white_data(this_rs_first,
                                                                                                    this_rs_last)

    logger.info("loading dataset finished")

    labeled_features_list.extend(unlabeled_features_list)
    labeled_list.extend(unlabeled_list)
    labeled_features_list = np.array(labeled_features_list)
    labeled_list = np.array(labeled_list)

    test_rs.extend(test_white)
    test_rs_label.extend(test_white_label)

    test_x = np.array(test_rs)
    test_y = np.array(test_rs_label)
    test_x = test_x.astype(np.float64)

    # train_x, train_y = labeled_features_list, labeled_list

    train_x, test_no_use_1, train_y, test_no_use_2 = train_test_split(labeled_features_list, labeled_list,
                                                                      test_size=0.01)

    logger.info(sum(train_y))
    bc = BaggingClassifierPU(
        DecisionTreeClassifier(),
        n_estimators=1000,  # 1000 trees as usual
        max_samples=sum(train_y),  # Balance the positives and unlabeled in each bag
        n_jobs=40  # Use all cores
    )
    logger.info('............')
    bc.fit(train_x, train_y)
    predict_results = bc.predict(test_x)
    logger.info('start check............')
    bitcoin_heist_test(predict_results, test_y)


def generic_detection(rs):
    logger.info("loading dataset......")

    labeled_features_list, labeled_list, test_x_1, test_y_1 = load_labeled_address_data(rs)
    this_rs_first, this_rs_last = get_one_rs_active_date(rs)
    unlabeled_features_list, unlabeled_list, test_x_2, test_y_2 = load_all_period_year_data(this_rs_first, this_rs_last)


    logger.info("loading dataset finished")

    labeled_features_list.extend(unlabeled_features_list)
    labeled_list.extend(unlabeled_list)
    labeled_features_list = np.array(labeled_features_list)
    labeled_list = np.array(labeled_list)

    labeled_features_list = labeled_features_list.astype(np.float64)

    test_x_1.extend(test_x_2)
    test_y_1.extend(test_y_2)
    test_x_1 = np.array(test_x_1)
    test_y_1 = np.array(test_y_1)
    test_x_1 = test_x_1.astype(np.float64)

    train_x, test_x_no_use, train_y, test_y_no_use = train_test_split(labeled_features_list, labeled_list, test_size=0.2)
    train_x_2, test_x_true, train_y_2, test_y_true = train_test_split(test_x_1, test_y_1, test_size=0.2)

    train_x_true = np.vstack((train_x, train_x_2))
    train_y_true = np.hstack((train_y, train_y_2))
    test_x = test_x_true
    test_y = test_y_true

    logger.info(sum(train_y_true))
    bc = BaggingClassifierPU(
        GradientBoostingClassifier(),
        n_estimators=1000,  # 1000 trees as usual
        max_samples=sum(train_y_true),  # Balance the positives and unlabeled in each bag
        n_jobs=40  # Use all cores
    )
    bc.fit(train_x_true, train_y_true)
    predict_results = bc.predict(test_x)

    bitcoin_heist_test(predict_results, test_y)


def bitcoin_heist_test(y_pred, y_true):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    length = len(y_pred)
    for i in range(length):
        if y_true[i] == 1 and y_pred[i] == 1:
            tp += 1
        if y_true[i] == 0 and y_pred[i] == 1:
            fp += 1
        if y_true[i] == 1 and y_pred[i] == 0:
            fn += 1
        if y_true[i] == 0 and y_pred[i] == 0:
            tn += 1

    logger.info(tp)
    logger.info(fp)
    logger.info(fn)
    logger.info(tn)

    logger.info('ac ' + str(cal_ac(tp, fp, tn, fn)))
    logger.info('pr ' + str(cal_pr(tp, fp)))
    logger.info('rec ' + str(rec(tp, fn)))
    logger.info('f1 ' + str(f1(tp, fp, fn)))


def cal_ac(tp, fp, tn, fn):
    return float((tp + tn) / (tp + tn + fp + fn))


def cal_pr(tp, fp):
    return float(tp / (tp + fp))


def rec(tp, fn):
    return float(tp / (tp + fn))


def f1(tp, fp, fn):
    return float(2 * tp / (2 * tp + fp + fn))


# 计算精确率和召回率：
def accuracy_calculation(y_pred, y_true):
    # 准确率
    logger.info("accuracy:")
    logger.info(accuracy_score(y_true, y_pred))
    # 精确率
    logger.info("macro precision:")
    logger.info(precision_score(y_true, y_pred, average='macro'))
    logger.info("micro precision:")
    logger.info(precision_score(y_true, y_pred, average='micro'))
    # macro召回率
    logger.info("macro_recall:")
    logger.info(recall_score(y_true, y_pred, average='macro'))
    # micro召回率
    logger.info("micro_recall:")
    logger.info(recall_score(y_true, y_pred, average='micro'))
    # f1_score
    logger.info("macro_f1_score")
    logger.info(f1_score(y_true, y_pred, average='macro'))
    logger.info("micro_f1_score")
    logger.info(f1_score(y_true, y_pred, average='micro'))


if __name__ == "__main__":
    rs_famliy = ['CryptoLocker', 'CryptoWall', 'CryptXXX', 'Locky', 'Cerber']
    for one_rs in rs_famliy:
        now_time = datetime.datetime.now()
        print(now_time)
        pu_learning_detection(one_rs)
        now_time_end = datetime.datetime.now()
        print(now_time_end)


    # rs_famliy = ['CryptoWall']
    # rs_famliy = ['CryptoWall', 'CryptXXX', 'Locky', 'Cerber', 'DMALocker']
    # for one_rs in rs_famliy:
    #     unknown_detection(one_rs)

    # need_to_detect = ['Locky', 'Cerber']
    # for item in need_to_detect:
    #     generic_detection(item)
