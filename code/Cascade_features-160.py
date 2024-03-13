import os
import sys
import csv
from loguru import logger
from typing import List
import rocksdb
from tqdm import tqdm
import time
from multiprocess import Process
import math

csv.field_size_limit(sys.maxsize)

addr_self_feature_path = "../addr_self_feature/addr_self_feature.db"
addr_oppos_path = "../addr_oppos/addr_oppos.db"

def cascade_feature_calculation(addr: str) -> List:
    addr_self_feature_db = rocksdb.DB(addr_self_feature_path, rocksdb.Options(create_if_missing=True), read_only=True)
    addr_oppos_db = rocksdb.DB(addr_oppos_path, rocksdb.Options(create_if_missing=True), read_only=True)
    addr_self_feature = addr_self_feature_db.get(bytes(addr, encoding='utf-8'))
    addr_self_feature = eval(str(addr_self_feature, encoding='utf-8'))
    addr_self_feature_length = len(addr_self_feature)

    addr_oppos = addr_oppos_db.get(bytes(addr, encoding='utf-8'))
    addr_oppos = eval(str(addr_oppos, encoding='utf-8'))

    for role in addr_oppos:
        feature_list = []
        for sole_addr in role:
            addr_feature = addr_self_feature_db.get(bytes(sole_addr, encoding='utf-8'))
            if addr_feature is None:
                continue
            addr_feature = eval(str(addr_feature, encoding='utf-8'))
            feature_list.append(addr_feature)
        
        if len(feature_list) == 0:
            for i in range(addr_self_feature_length):
                for j in range(4):
                    addr_self_feature.append(0.0)
            continue
            
        for i in range(addr_self_feature_length):
            one_features_list = [float(one_features[i]) for one_features in feature_list]
            max_value, min_value, mean_value, std_value = compute_list_attribute(one_features_list)
            addr_self_feature.append(max_value)
            addr_self_feature.append(min_value)
            addr_self_feature.append(mean_value)
            addr_self_feature.append(std_value)
    
    addr_self_feature.insert(0, addr)
    
    return addr_self_feature


def compute_list_attribute(list):
    max_value = list[0]
    min_value = list[0]
    mean_value = list[0]
    std_value = list[0]
    sum_value = 0
    for item in list:
        sum_value += item
        if item > max_value:
            max_value = item
        if item < min_value:
            min_value = item

    mean_value = sum_value / len(list)

    sqrt_value = 0
    for item in list:
        sqrt_value += (item - mean_value) * (item - mean_value)
    std_value = math.sqrt(sqrt_value)

    return max_value, min_value, mean_value, std_value



def compute_ransomware_cascade_features(addr_list: list, pid: int):
    writer = csv.writer(open("../addr_cascade_feature/ransomware_addr/"+ str(pid) + ".csv", "a+"))
    for addr in tqdm(addr_list):
        cascade_features = cascade_feature_calculation(addr)
        writer.writerow(cascade_features)


def compute_ransomware_cascade_features_mul_process():
    ransomware_addr = list()
    reader = csv.reader(open("../ransomware_dataset/final_ransomware_address.csv", "r"))
    for row in reader:
        ransomware_addr.append(row[0])

    one_process_length = len(ransomware_addr) // 59
    pid_list = []
    for i in range(60):
        pid_list.append(Process(target=compute_ransomware_cascade_features, args=(ransomware_addr[i*one_process_length: (i+1) * one_process_length],i, )))
    
    for p in pid_list:
        p.start()
    
    for p in pid_list:
        p.join()


def compute_random_1_cascade_features(addr_list: list, pid: int):
    writer = csv.writer(open("../addr_cascade_feature/random_addr_1/"+ str(pid) + ".csv", "a+"))
    for addr in tqdm(addr_list):
        cascade_features = cascade_feature_calculation(addr)
        writer.writerow(cascade_features)


def compute_random_1_cascade_features_mul_process():
    ransomware_addr = list()
    reader = csv.reader(open("../random_addr_1000000_1.csv", "r"))
    for row in reader:
        ransomware_addr.append(row[0])
    
    ransomware_addr = ransomware_addr[333333:666666]

    one_process_length = len(ransomware_addr) // 39
    pid_list = []
    for i in range(40):
        pid_list.append(Process(target=compute_random_1_cascade_features, args=(ransomware_addr[i*one_process_length: (i+1) * one_process_length],i, )))
    
    for p in pid_list:
        p.start()
    
    for p in pid_list:
        p.join()

def compute_random_2_cascade_features(addr_list: list, pid: int):
    writer = csv.writer(open("../addr_cascade_feature/random_addr_2_160/"+ str(pid) + ".csv", "a+"))
    for addr in tqdm(addr_list):
        cascade_features = cascade_feature_calculation(addr)
        writer.writerow(cascade_features)


def compute_random_2_cascade_features_mul_process():
    ransomware_addr = list()
    reader = csv.reader(open("../random_addr_1000000_2.csv", "r"))
    for row in reader:
        ransomware_addr.append(row[0])
    
    ransomware_addr = ransomware_addr[333333:666666]

    one_process_length = len(ransomware_addr) // 39
    pid_list = []
    for i in range(40):
        pid_list.append(Process(target=compute_random_2_cascade_features, args=(ransomware_addr[i*one_process_length: (i+1) * one_process_length],i, )))
    
    for p in pid_list:
        p.start()
    
    for p in pid_list:
        p.join()


def compute_random_entity_cascade_features(addr_list: list, pid: int):
    writer = csv.writer(open("../addr_cascade_feature/random_entity_addr_160/"+ str(pid) + ".csv", "a+"))
    for addr in tqdm(addr_list):
        cascade_features = cascade_feature_calculation(addr)
        writer.writerow(cascade_features)


def compute_random_entity_cascade_features_mul_process():
    ransomware_addr = list()
    reader = csv.reader(open("../random_entity_addr.csv", "r"))
    for row in reader:
        ransomware_addr.append(row[0])
    
    ransomware_addr = ransomware_addr[333333:666666]

    one_process_length = len(ransomware_addr) // 39
    pid_list = []
    for i in range(40):
        pid_list.append(Process(target=compute_random_entity_cascade_features, args=(ransomware_addr[i*one_process_length: (i+1) * one_process_length],i, )))
    
    for p in pid_list:
        p.start()
    
    for p in pid_list:
        p.join()
    
    # compute_ransomware_cascade_features(ransomware_addr, 0)

if __name__ == "__main__":
    start_time = time.time()
    logger.info("start")
    # compute_random_2_cascade_features_mul_process()
    compute_random_entity_cascade_features_mul_process()
    logger.info("finish")
    end_time = time.time()
    logger.info(end_time - start_time)

