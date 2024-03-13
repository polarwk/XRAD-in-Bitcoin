import os
import sys
import csv
from typing import List
import random
import numpy as np
from baggingPU import BaggingClassifierPU
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from loguru import logger
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# 加载指定勒索家族的级联特征
def load_specific_rs(name: str) -> List:
    reader = csv.reader(
        open("../addr_cascade_feature/ransomware_addr_cascade_feature_with_label.csv"))
    feature_list = []
    for row in reader:
        if row[1] == name:
            feature_list.append(row[2:])

    return feature_list


# 生成一个随机数集合，代表在文件中的index
def random_white(addr_num: int):
    addr_index = set()
    random.seed(time.time())
    while len(addr_index) < addr_num:
        index = random.randint(0, 900000)
        addr_index.add(index)

    return addr_index


# 加载指定数量的随机白样本地址
def load_white_addresses(addr_num: int):
    random_addr_index = random_white(addr_num)
    reader = csv.reader(
        open(
            "../addr_cascade_feature/random_addr_1_cascade_feature_filtered_remove.csv",
            "r"))

    feature_list = []
    index = 0
    for row in reader:
        if index in random_addr_index:
            feature_list.append(row[1:])

        index += 1

    return feature_list


# 使用PU_learning进行实验一
def specific_rs_detection(rs_name: str):
    rs_features = load_specific_rs(rs_name)
    white_features = load_white_addresses(100000)

    rs_label = [1 for i in range(len(rs_features))]
    white_label = [0 for i in range(len(white_features))]
    logger.info("load data finish")

    rs_features.extend(white_features)
    features = np.array(rs_features)
    features = features.astype(np.float64)
    rs_label.extend(white_label)
    labels = np.array(rs_label)
    labels = labels.astype(np.float64)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, shuffle=True)
    logger.info("max_samples:" + str(int(sum(y_train))))
    bc = BaggingClassifierPU(
        DecisionTreeClassifier(),
        n_estimators=10,  # 1000 trees as usual
        max_samples=int(sum(y_train)),  # Balance the positives and unlabeled in each bag
        n_jobs=40  # Use all cores
    )
    bc.fit(X_train, y_train)

    pred = bc.predict(X_test)
    accuracy_calculation(pred, y_test)


# 提取其他勒索家族和当前勒索家族的特征
def load_other_rs_addrs_from_specific(rs_name: str):
    others_features = list()
    spec_features = list()
    reader = csv.reader(
        open(
            "../addr_cascade_feature/final_ransomware_addr_cascade_feature_with_label_remove.csv"))
    for row in reader:
        if row[1] == rs_name:
            spec_features.append(row[2:])
        else:
            others_features.append(row[2:])
    return spec_features, others_features


# 使用其他所有勒索家族的地址来检测当前勒索家族地址
def generic_rs_detection(rs_name: str):
    logger.info(rs_name)
    spec_features, other_features = load_other_rs_addrs_from_specific(rs_name)
    white_features = load_white_addresses(100000)

    other_white_features = white_features[0: 90000]
    spec_white_features = white_features[90000:]

    other_label = [1 for i in range(len(other_features))]
    other_white_label = [0 for i in range(len(other_white_features))]

    spec_label = [1 for i in range(len(spec_features))]
    spec_white_label = [0 for i in range(len(spec_white_features))]
    logger.info("load data finish")

    other_features.extend(other_white_features)
    train_features = np.array(other_features)
    train_features = train_features.astype(np.float64)
    other_label.extend(other_white_label)
    train_label = np.array(other_label)
    train_label = train_label.astype(np.float64)

    np.random.seed(1)
    np.random.shuffle(train_features)
    np.random.seed(1)
    np.random.shuffle(train_label)

    logger.info("max_samples:" + str(int(sum(train_label))))

    bc = BaggingClassifierPU(
        DecisionTreeClassifier(),
        n_estimators=100,  # 1000 trees as usual
        max_samples=int(sum(train_label)),  # Balance the positives and unlabeled in each bag
        n_jobs=-1,  # Use all cores
        max_features=4
    )
    bc.fit(train_features, train_label)

    spec_features.extend(spec_white_features)
    spec_features = np.array(spec_features)
    spec_features = spec_features.astype(np.float64)
    spec_label.extend(spec_white_label)
    spec_label = np.array(spec_label)
    spec_label = spec_label.astype(np.float64)

    pred = bc.predict(spec_features)
    accuracy_calculation(pred, spec_label)


# 加载每个rs的活跃时间段
def load_rs_active_period():
    active_period_path = "../ransomware_dataset/ransomware_addr_active_time_period.csv"
    active_period = dict()
    reader = csv.reader(open(active_period_path, "r"))
    for row in reader:
        active_period[row[0]] = [row[1], row[2]]
    return active_period


# 加载当前rs和之前发生的rs
def load_other_rs_before_specific(rs_name: str):
    active_period = load_rs_active_period()
    start_time = active_period[rs_name][0]

    before_rs_list = []

    for label, period in active_period.items():
        if label == rs_name:
            continue

        if period[0] < start_time:
            before_rs_list.append(label)

    others_features = list()
    spec_features = list()
    reader = csv.reader(
        open(
            "../addr_cascade_feature/ransomware_addr_cascade_feature_with_label_extra.csv"))
    for row in reader:
        if row[1] == rs_name:
            spec_features.append(row[2:])
        elif row[1] in before_rs_list:
            others_features.append(row[2:])
    return spec_features, others_features


# 进行Unknown ransomware family检测
def unknown_rs_detection(rs_name: str):
    spec_features, other_features = load_other_rs_before_specific(rs_name)
    white_features = load_white_addresses(100000)

    other_white_features = white_features[0: int(0.7 * len(white_features))]
    spec_white_features = white_features[int(0.7 * len(white_features)):]

    other_label = [1 for i in range(len(other_features))]
    other_white_label = [0 for i in range(len(other_white_features))]

    spec_label = [1 for i in range(len(spec_features))]
    spec_white_label = [0 for i in range(len(spec_white_features))]
    logger.info("load data finish")

    other_features.extend(other_white_features)
    train_features = np.array(other_features)
    train_features = train_features.astype(np.float64)
    other_label.extend(other_white_label)
    train_label = np.array(other_label)
    train_label = train_label.astype(np.float64)

    np.random.seed(1)
    np.random.shuffle(train_features)
    np.random.seed(1)
    np.random.shuffle(train_label)

    logger.info("max_samples:" + str(int(sum(train_label))))

    bc = BaggingClassifierPU(
        DecisionTreeClassifier(),
        n_estimators=100,  # 1000 trees as usual
        max_samples=int(sum(train_label)),  # Balance the positives and unlabeled in each bag
        n_jobs=40  # Use all cores
    )
    bc.fit(train_features, train_label)

    spec_features.extend(spec_white_features)
    spec_features = np.array(spec_features)
    spec_features = spec_features.astype(np.float64)
    spec_label.extend(spec_white_label)
    spec_label = np.array(spec_label)
    spec_label = spec_label.astype(np.float64)

    pred = bc.predict(spec_features)
    accuracy_calculation(pred, spec_label)


# 计算精确率和召回率：
def accuracy_calculation(y_pred, y_true):
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(y_pred)):
        if y_pred[i] == 1:
            if y_true[i] == 1:
                TP += 1
            else:
                FP += 1
        else:
            if y_true[i] == 1:
                FN += 1
            else:
                TN += 1
    logger.info("TP:" + str(TP) + ", FP:" + str(FP) + ", TN:" + str(TN) + ", FN:" + str(FN))
    # 准确率
    logger.info("accuracy:")
    logger.info(accuracy_score(y_true, y_pred))
    # 精确率
    logger.info("macro precision:")
    logger.info(precision_score(y_true, y_pred, average='macro'))
    logger.info("micro precision:")
    logger.info(precision_score(y_true, y_pred, average='micro'))
    logger.info("precision:")
    logger.info(precision_score(y_true, y_pred))

    # macro召回率
    logger.info("macro_recall:")
    logger.info(recall_score(y_true, y_pred, average='macro'))
    # micro召回率
    logger.info("micro_recall:")
    logger.info(recall_score(y_true, y_pred, average='micro'))
    logger.info("recall:")
    logger.info(recall_score(y_true, y_pred))

    # f1_score
    logger.info("macro_f1_score")
    logger.info(f1_score(y_true, y_pred, average='macro'))
    logger.info("micro_f1_score")
    logger.info(f1_score(y_true, y_pred, average='micro'))
    logger.info("f1_score")
    logger.info(f1_score(y_true, y_pred))


if __name__ == "__main__":
    # specific_rs_detection("CryptoLocker")
    generic_rs_detection("CryptoLocker")
    # unknown_rs_detection("DMA-Locker")
