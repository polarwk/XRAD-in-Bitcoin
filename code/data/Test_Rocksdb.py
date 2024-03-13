import os
import csv
csv.field_size_limit(1024*1024*1024)
from loguru import logger
import rocksdb

def test_addr_feature():
    addr_feature_db_path = "../addr_feature_new.db"
    addr_feature_db = rocksdb.DB(addr_feature_db_path, rocksdb.Options(create_if_missing=True), read_only=True)

    white_addr_oppps_path = "../nfs/wk/TDSC/white_address_oppos/year_random_white_oppos.csv"
    white_addr_oppps_f = open(white_addr_oppps_path, "r")
    white_addr_oppps_reader = csv.reader(white_addr_oppps_f)


    not_finish_oppos_addr_path = "../nfs/wk/TDSC/white_address_oppos/not_finish_white_oppos.csv"
    not_finish_oppos_addr_f= open(not_finish_oppos_addr_path, "a+")
    not_finish_oppos_addr_writer = csv.writer(not_finish_oppos_addr_f)
    
    count = 0
    for row in white_addr_oppps_reader:
        if count % 100000 == 0:
            logger.info(count)
        count += 1
        result = addr_feature_db.get(bytes(row[0], encoding='utf-8'))
        if result is None:
            not_finish_oppos_addr_writer.writerow([row[0]])


def test_ransom_oppos_feature():
    addr_feature_db_path = "../addr_feature_new.db"
    addr_feature_db = rocksdb.DB(addr_feature_db_path, rocksdb.Options(create_if_missing=True), read_only=True)

    white_addr_oppps_path = "../nfs/wk/TDSC/ransom_address_oppos/all_oppos.csv"
    white_addr_oppps_f = open(white_addr_oppps_path, "r")
    white_addr_oppps_reader = csv.reader(white_addr_oppps_f)


    not_finish_oppos_addr_path = "../nfs/wk/TDSC/ransom_address_oppos/not_finish_ransom_oppos.csv"
    not_finish_oppos_addr_f= open(not_finish_oppos_addr_path, "a+")
    not_finish_oppos_addr_writer = csv.writer(not_finish_oppos_addr_f)
    
    count = 0
    for row in white_addr_oppps_reader:
        if count % 100000 == 0:
            logger.info(count)
        count += 1
        result = addr_feature_db.get(bytes(row[0], encoding='utf-8'))
        if result is None:
            logger.info(row[0])
            # not_finish_oppos_addr_writer.writerow([row[0]])


if __name__ == "__main__":
    test_ransom_oppos_feature()

