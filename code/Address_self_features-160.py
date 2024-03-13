import os
import sys
import csv
from loguru import logger
import blocksci
from multiprocess import Process
from typing import Set, List
import time
import random
import rocksdb
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

blockchain = blocksci.Blockchain("../BlockSci-0.7.0/config.json")
# self_feature_path = "../addr_self_feature/addr_self_feature.db"
# self_feature_db = rocksdb.DB(self_feature_path, rocksdb.Options(create_if_missing=True))


def get_all_mixTransactions() -> Set[str]:
    path = "../mixingTx.csv"
    reader = csv.reader(open(path, "r"))
    mixTransactions = set()
    for row in reader:
        mixTransactions.add(row[0])

    return mixTransactions


def extract_address_features(addr: str) -> List:
    transaction_count = 0
    # 金额上
    receive_transaction_count = 0  # 接收多少次
    receive_transaction_amount = 0  # 接收总金额
    avg_receive_amount = 0  # 平均每次接收金额
    max_receive_amount = 0  # 最大接收金额
    min_receive_amount = float('inf')  # 最小接收金额
    send_transaction_count = 0  # 发送多少次
    send_transaction_amount = 0  # 发送总金额
    avg_send_amount = 0  # 平均每次发送金额
    max_send_amount = 0  # 最大发送金额
    min_send_amount = float('inf')  # 最小发送金额
    balance = 0  # 余额

    # 交易输入输出个数
    max_inputs_count = 0  # 最大交易输入个数
    min_inputs_count = sys.maxsize  # 最小交易输入个数
    max_outputs_count = 0  # 最大交易输出个数
    min_outputs_count = sys.maxsize  # 最小交易输出个数

    # 时间上
    has_locktime = 0  # 参与的交易中有无设置locktime
    active_period = 0  # 地址参与交易的时间段

    # 混币交易
    has_mixing = 0  # 是否参与过混币交易

    # 开始提取
    addr_object = blockchain.address_from_string(addr)
    time_list = addr_object.txes.block_height
    if len(time_list) == 0:
        return None
    start_time = min(time_list)
    end_time = max(time_list)
    active_period = end_time - start_time

    receive_transaction_count = addr_object.output_txes_count()
    send_transaction_count = addr_object.input_txes_count()
    transaction_count = receive_transaction_count + send_transaction_count

    balance = addr_object.balance()

    for output in addr_object.outputs:
        receive_transaction_amount += output.value
        if min_receive_amount > output.value:
            min_receive_amount = output.value
        if max_receive_amount < output.value:
            max_receive_amount = output.value

    if receive_transaction_count != 0 and receive_transaction_amount != 0:
        avg_receive_amount = receive_transaction_amount / receive_transaction_count

    for input in addr_object.inputs:
        send_transaction_amount += input.value
        if min_send_amount > input.value:
            min_send_amount = input.value
        if max_send_amount < input.value:
            max_send_amount = input.value
    if send_transaction_count != 0 and send_transaction_amount != 0:
        avg_send_amount = send_transaction_amount / send_transaction_count

    for tx in addr_object.txes:
        if max_inputs_count < tx.input_count:
            max_inputs_count = tx.input_count
        if min_inputs_count > tx.input_count:
            min_inputs_count = tx.input_count

        if max_outputs_count < tx.output_count:
            max_outputs_count = tx.output_count
        if min_outputs_count > tx.output_count:
            min_outputs_count = tx.output_count
        if tx.locktime > 0:
            has_locktime = 1
        if str(tx.hash) in mixTransactions:
            has_mixing = 1

    return [addr, transaction_count, receive_transaction_count, receive_transaction_amount,
            avg_receive_amount, max_receive_amount, min_receive_amount, send_transaction_count,
            send_transaction_amount, avg_send_amount, max_send_amount, min_send_amount, max_inputs_count,
            min_inputs_count, max_outputs_count, min_outputs_count, has_mixing, has_locktime, active_period]


def extract_ransomware_addr_list_feature(addr_list):
    for addr in addr_list:
        extract_address_features(addr)


def extract_addr_feature_test():
    ransomware_address_path = "../ransomware_dataset/final_ransomware_address.csv"
    reader = csv.reader(open(ransomware_address_path, "r"))
    process_list = []
    addr_list = []
    for row in reader:
        addr = row[0]
        addr_list.append(addr)

    length = len(addr_list) // 10
    for i in range(10):
        pid = Process(target=extract_ransomware_addr_list_feature, args=(addr_list[i * length: (i + 1) * length],))
        process_list.append(pid)

    for pid in process_list:
        pid.start()

    for pid in process_list:
        pid.join()


def extract_ransomware_addr_feature():
    ransomware_address_path = "../ransomware_dataset/final_ransomware_address.csv"
    reader = csv.reader(open(ransomware_address_path, "r"))
    writer = csv.writer(
        open("../addr_self_feature/ransomware_address_features.csv", "a+"))
    for row in reader:
        addr = row[0]
        feature = extract_address_features(addr)
        writer.writerow(feature)


def extract_white_addr_feature_one_process(addr_list, pid):
    writer = csv.writer(open("../addr_self_feature/random_entity_addr_oppos-160/" + str(pid) + ".csv", "a+"))
    for addr in tqdm(addr_list):
        features = extract_address_features(addr)
        if features is None:
            logger.info(addr)
            continue
        writer.writerow(features)


def load_address_extract_self_features():
    addr_set = set()
    file_path_list = ["../random_addr_1000000_1.csv", 
                      "../addr_oppos/random_addr_1_oppos_combined.csv",
                      "../ransomware_dataset/final_ransomware_address.csv", 
                      "../random_addr_1000000_2.csv",
                      "../addr_oppos/random_addr_2_oppos_combined.csv"]
    for file_path in file_path_list:
        reader = csv.reader(open(file_path, "r"))
        for row in reader:
            addr_set.add(row[0])
    return addr_set

def extract_white_addr_feature_multi_process():
    allready_address = load_address_extract_self_features()
    path = "../addr_oppos/random_entity_addr_oppos_combined.csv"
    reader = csv.reader(open(path, "r"))
    addr_list = []
    for row in reader:
        addr_list.append(row[0])
    
    addr_list = set(addr_list)
    addr_list = addr_list - allready_address
    addr_list = list(addr_list)
    logger.info(len(addr_list))

    # addr_list = addr_list[0:12710000]
    
    process_length = len(addr_list) // 39
    p_list = []
    for i in range(40):
        p_list.append(Process(target=extract_white_addr_feature_one_process, args=(addr_list[i*process_length: (i+1)*process_length],i,)))
    
    for p in p_list:
        p.start()
    
    for p in p_list:
        p.join()



def upload_self_features_rocksdb():
    file_f = open("../addr_self_feature/ransomware_address_features.csv", "r")
    file_reader = csv.reader(file_f)
    for row in file_reader:
        addr = row[0]
        feature = row[1:]
        self_feature_db.put(bytes(addr, encoding='utf-8'), bytes(str(feature), encoding='utf-8'))


def collectionMethod(n):
    """集合方式实现"""
    numbers = set()
    while len(numbers) < n:
        # 0~8亿之间的随机数
        i = random.randint(0, 800000000)
        numbers.add(i)
    return numbers


def sample_white_addresses(addr_num: int):
    random_addr_index = collectionMethod(addr_num)
    random_address = set()
    for addr_index in random_addr_index:
        addr_object = blockchain.address_from_index(addr_index)


def sample_white_addresses_in_string(addr_num: int):
    path = "../update_to_2022_12_30.csv"
    random_num = collectionMethod(addr_num + 100000)
    logger.info("random num finish")
    random_addr = set()
    reader = csv.reader(open(path, "r"), delimiter=";")
    index = 0
    for row in reader:
        addr = row[0]
        if index in random_num:
            addr_object = blockchain.address_from_string(addr)
            if addr_object is not None and len(addr_object.txes.to_list()) > 0:
                random_addr.add(addr)
                if len(random_addr) == addr_num:
                    break

        index += 1
    
    logger.info("random addr finish")
    
    write_path = "../random_addr_1000000_1.csv"
    writer = csv.writer(open(write_path, "a+"))
    for addr in random_addr:
        writer.writerow([addr])
    
    return random_addr
        



if __name__ == '__main__':
    start_time = time.time()
    # test_blocksci_addr("12QxqCNFrxYG8Qcawyng1GwN6Crapgkxx3")
    mixTransactions = get_all_mixTransactions()
    extract_white_addr_feature_multi_process()
    # upload_self_features_rocksdb()
    # sample_white_addresses_in_string(1000000)
    end_time = time.time()
    logger.info(end_time - start_time)
    
