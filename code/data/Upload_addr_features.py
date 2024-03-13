import os
import csv
csv.field_size_limit(1024*1024*1024)
import rocksdb
from loguru import logger

addr_feature_db_path = "../addr_feature_new.db"
addr_feature_db = rocksdb.DB(addr_feature_db_path, rocksdb.Options(create_if_missing=True))

def update_white_addr_oppo_features():
    white_oppo_features_path = "../addr_oppos_feature/white/"
    file_list = os.listdir(white_oppo_features_path)
    file_list = sorted(file_list)
    for file in file_list:
        logger.info(file)
        file_path = white_oppo_features_path + file
        file_f = open(file_path, "r")
        file_reader = csv.reader(file_f)
        for row in file_reader:
            addr = row[0]
            feature = row[1:]
            addr_feature_db.put(bytes(addr, encoding='utf-8'), bytes(str(feature), encoding='utf-8'))

def upload_white_addr_features():
    white_addr_features_path = "../nfs/wk/TDSC/addr_self_feature/white/"
    for year in range(2013, 2021):
        year = str(year)
        logger.info(year)
        year_file_path = white_addr_features_path + year + ".csv"
        year_file_f = open(year_file_path, "r")
        year_file_reader = csv.reader(year_file_f)
        for row in year_file_reader:
            if row[0] == "address":
                continue
            addr = row[0]
            feature = row[1:]
            addr_feature_db.put(bytes(addr, encoding='utf-8'), bytes(str(feature), encoding='utf-8'))

def upload_over_length_addr():
    file_path = "../addr_oppos_feature/white/year_random_addr_oppos_feature04.csv"
    file_f = open(file_path, "r")
    file_reader = csv.reader(file_f)
    for row in file_reader:
        addr = row[0]
        feature = row[1:]
        addr_feature_db.put(bytes(addr, encoding='utf-8'), bytes(str(feature), encoding='utf-8'))

def upload_ransom_self_feature():
    file_path = "../nfs/wk/TDSC/addr_self_feature/ransom/all_ransom_feature.csv"
    file_f = open(file_path, "r")
    file_reader = csv.reader(file_f)
    count = 0
    for row in file_reader:
        addr = row[0]
        logger.info(count)
        count += 1
        feature = row[1:]
        addr_feature_db.put(bytes(addr, encoding='utf-8'), bytes(str(feature), encoding='utf-8'))

def upload_ransom_oppos_feature():
    ransom_oppo_features_path = "../nfs/wk/TDSC/addr_oppos_feature/ransom/"
    file_list = os.listdir(ransom_oppo_features_path)
    file_list = sorted(file_list)
    for file in file_list:
        logger.info(file)
        file_path = ransom_oppo_features_path + file
        file_f = open(file_path, "r")
        file_reader = csv.reader(file_f)
        for row in file_reader:
            addr = row[0]
            feature = row[1:]
            addr_feature_db.put(bytes(addr, encoding='utf-8'), bytes(str(feature), encoding='utf-8'))

def upload_not_finish_ransom_addr_feature():
    file_path = "../nfs/wk/TDSC/addr_oppos_feature/ransom/not_finish_addr.csv"
    file_f = open(file_path, "r")
    file_reader = csv.reader(file_f)
    count = 0
    for row in file_reader:
        addr = row[0]
        logger.info(count)
        count += 1
        feature = row[1:]
        addr_feature_db.put(bytes(addr, encoding='utf-8'), bytes(str(feature), encoding='utf-8'))


if __name__ == "__main__":
    upload_ransom_self_feature()
