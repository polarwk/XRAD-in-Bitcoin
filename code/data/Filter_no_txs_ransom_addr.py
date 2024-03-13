import os
import csv
from loguru import logger
import rocksdb

addr_txs_path = "../cd_data/tsdc/addr_related_txs.db"
addr_txs_db = rocksdb.DB(addr_txs_path, rocksdb.Options(create_if_missing=True))

def filter_no_txs_ransom_addr():
    origin_addr_path = "../nfs/wk/TDSC/24486_addrs.csv"
    origin_addr_f = open(origin_addr_path)
    origin_addr_reader = csv.reader(origin_addr_f)

    has_txs_addr_path = "../nfs/wk/TDSC/24440_addrs.csv"
    has_txs_addr_f = open(has_txs_addr_path, "a+")
    has_txs_addr_writer = csv.writer(has_txs_addr_f)

    for row in origin_addr_reader:
        addr = row[0]
        result = addr_txs_db.get(bytes(addr, encoding='utf-8'))
        if result is not None:
            has_txs_addr_writer.writerow([addr])

if __name__ == "__main__":
    filter_no_txs_ransom_addr()
