import os
import csv
import sys
from loguru import logger
import re
csv.field_size_limit(sys.maxsize)

## 固定长度拆分字符串
def cut_text(text,lenth):
    textArr = re.findall(r'.{'+str(lenth)+'}', text)
    textArr.append(text[(len(textArr)*lenth):])
    if(len(text)%lenth==0):
        return textArr[0:-1]
    else:
        return textArr

def split_hash():
    folder = "../nfs/hyw/crawl_data_from_walletexploer/crawl_labeled_transaction_hash/datas1/"
    new_folder = "../nfs/hyw/crawl_data_from_walletexploer/crawl_labeled_transaction_hash/datas1_new/"
    file_list = os.listdir(folder)
    for file in file_list:
        writer = csv.writer(open(new_folder + file, "a+"))
        file_path = folder + file
        reader = csv.reader(open(file_path, "r"))
        for row in reader:
            hashes = row[0]
            hash_list = cut_text(hashes, 64)
            for hash in hash_list:
                writer.writerow([hash])

def combine_mixing_transactions():
    paths = ["../nfs/hyw/crawl_data_from_walletexploer/crawl_labeled_transaction_hash/datas/", "../nfs/hyw/crawl_data_from_walletexploer/crawl_labeled_transaction_hash/datas1_new/", "../nfs/wk/TDSC/mixing_transactions/"]
    mixing_transactions = set()
    for path in paths:
        file_list = os.listdir(path)
        for file in file_list:
            file_path = path + file
            reader = csv.reader(open(file_path, "r"))
            for row in reader:
                mixing_transactions.add(row[0])
    
    mixing_tx_path1 = "../nfs/wk/MonthlyReport/cluster_preprocess/txid_full.csv"
    reader = csv.reader(open(mixing_tx_path1, "r"))
    for row in reader:
        mixing_transactions.add(row[0])
    
    combined_mixing_tx_path = "../mixingTx.csv"
    writer = csv.writer(open(combined_mixing_tx_path, "a+"))
    for tx in mixing_transactions:
        writer.writerow([tx])
    





if __name__ == "__main__":
    combine_mixing_transactions()
