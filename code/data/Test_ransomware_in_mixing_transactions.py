import os
import csv
from tkinter import Canvas
from loguru import logger
import rocksdb

addr_txs_path = "../addr_related_txs.db"
addr_txs_db = rocksdb.DB(addr_txs_path, rocksdb.Options(create_if_missing=True), read_only=True)

def get_mixing_transactions():

    mixing_transaction_set = set()
    mixing_transaction_path = "../nfs/wk/TDSC/mixing_transactions/"
    mixing_file_list = os.listdir(mixing_transaction_path)
    for mixing_file in mixing_file_list:
        mixing_file_path = mixing_transaction_path + mixing_file
        mixing_file_f = open(mixing_file_path, "r")
        mixing_file_reader = csv.reader(mixing_file_f)
        for row in mixing_file_reader:
            mixing_transaction_set.add(row[0])

    return mixing_transaction_set

def test_every_ransomware_activities():
    mixing_transaction_set = get_mixing_transactions()
    tag_path = "../nfs/wk/TDSC/addr_cascade_feature/ransom_new/tag/"
    tag_list = os.listdir(tag_path)
    for tag in tag_list:
        logger.info(tag)
        count = 0
        tag_file_path = tag_path + tag
        tag_file_f = open(tag_file_path, "r")
        tag_file_reader = csv.reader(tag_file_f)
        for row in tag_file_reader:
            addr = row[0]
            related_txs = addr_txs_db.get(bytes(addr, encoding='utf-8'))
            related_txs = str(related_txs, encoding='utf-8')
            related_txs = eval(related_txs)
            for tx in related_txs:
                if tx in mixing_transaction_set:
                    count += 1
                    break
        logger.info(count)

if __name__ == "__main__":
    test_every_ransomware_activities()

