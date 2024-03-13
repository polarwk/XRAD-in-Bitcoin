import os
import csv
csv.field_size_limit(1024*1024*1024)
import rocksdb
from loguru import logger

addr_cascade_feature_db_path = "../addr_cascade_feature_long.db"
addr_cascade_feature_db = rocksdb.DB(addr_cascade_feature_db_path, rocksdb.Options(create_if_missing=True))



def upload_white_addr_cascade():
    path = "../nfs/wk/TDSC/addr_cascade_feature/white_new/"
    file_list = os.listdir(path)
    for file in file_list:
        logger.info(file)
        file_path = path + file
        file_f = open(file_path, "r")
        file_reader = csv.reader(file_f)
        for row in file_reader:
            addr = row[0]
            cascade_feature = row[1:]
            addr_cascade_feature_db.put(bytes(addr, encoding='utf-8'), bytes(str(cascade_feature), encoding='utf-8'))

def upload_ransom_addr_cascade():
    logger.info("upload_ransom_addr_cascade")
    path = "../nfs/wk/TDSC/addr_cascade_feature/ransom_new_long/ransom_cascade_feature.csv"
    path_f = open(path, "r")
    path_reader = csv.reader(path_f)
    for row in path_reader:
        addr = row[0]
        cascade_feature = row[1:]
        addr_cascade_feature_db.put(bytes(addr, encoding='utf-8'), bytes(str(cascade_feature), encoding='utf-8'))


if __name__ == "__main__":
    upload_ransom_addr_cascade()



