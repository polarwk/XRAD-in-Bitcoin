# 利用之前做好的addr_features.db, 如果库中已经有了，并且特征中的交易数量也是对的，就直接用这个库中的特征作为地址特征，不过要新加上一个新特征：是否参与了混币交易。
import os
import csv
import json
csv.field_size_limit(1024*1024*1024)
from loguru import logger
import rocksdb
import sys
from multiprocessing import Pool

addr_txs_path = "../addr_related_txs.db"
addr_txs_db = rocksdb.DB(addr_txs_path, rocksdb.Options(create_if_missing=True), read_only=True)
tx_detail_path = "../bitcoin_transaction.db"
tx_detail_db = rocksdb.DB(tx_detail_path, rocksdb.Options(create_if_missing=True), read_only=True)
old_addr_feature_path = "../addr_feature.db"
old_addr_feature_db = rocksdb.DB(old_addr_feature_path, rocksdb.Options(create_if_missing=True), read_only=True)




def extract_white_address_features(one_addr):
    # f = open("../addr_oppos_feature/white/" + str(p_id) + ".csv", "a+")
    # csv_writer = csv.writer(f)
    # csv_writer.writerow(
    #     ["address", "transaction_count", "receive_transaction_count", "receive_transaction_amount",
    #      "avg_receive_amount",
    #      "max_receive_amount", "min_receive_amount", "send_transaction_count", "send_transaction_amount",
    #      "avg_send_amount", "max_send_amount", "min_send_amount", "max_inputs_count", "min_inputs_count",
    #      "max_outputs_count", "min_outputs_count", "has_mix", "has_locktime", "active_period"])
    # write_features_list = []

    transaction_dict_list = []

    # 混币交易
    has_mixing = 0

    addr_related_txs = addr_txs_db.get(bytes(one_addr, encoding='utf-8'))
    if addr_related_txs is not None:
        addr_related_txs = str(addr_related_txs, encoding='utf-8')
        addr_related_txs = eval(addr_related_txs)
        # if len(addr_related_txs) > 1000000:
            # logger.info("overlength:" + one_addr)
            # return [one_addr, len(addr_related_txs)]
        for this_tx_hash in addr_related_txs:
            if this_tx_hash in mixing_transaction_set:
                has_mixing = 1
            tx_detail_info = tx_detail_db.get(bytes(this_tx_hash, encoding='utf-8'))
            if tx_detail_info is not None:
                tx_detail_info = json.loads(str(tx_detail_info, encoding='utf-8'))
                transaction_dict_list.append(tx_detail_info)

    transaction_count = len(transaction_dict_list)
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

    # 交易输入输出个数
    max_inputs_count = 0  # 最大交易输入个数
    min_inputs_count = sys.maxsize  # 最小交易输入个数
    max_outputs_count = 0  # 最大交易输出个数
    min_outputs_count = sys.maxsize  # 最小交易输出个数


    # 时间上
    has_locktime = 0  # 参与的交易中有无设置locktime
    active_period = 0  # 地址参与交易的时间段

    transaction_time_list = []

    for transaction in transaction_dict_list:

        transaction_time_list.append(int(transaction["timestamp"]))

        inputs = transaction["vin"]
        outputs = transaction["vout"]
        send_value = 0

        n_in = len(inputs)
        n_out = len(outputs)
        # 更新关于交易输入输出的特征
        if n_in > max_inputs_count:
            max_inputs_count = n_in
        if n_in < min_inputs_count:
            min_inputs_count = n_in

        # 更新locktime
        if transaction["locktime"] != 0:
            has_locktime = 1

        if n_out > max_outputs_count:
            max_outputs_count = n_out
        if n_out < min_outputs_count:
            min_outputs_count = n_out

        for one_input in inputs:
            if one_input.get("address") == one_addr:
                send_transaction_count += 1
                one_send_value = float(one_input["value"])
                send_value += one_send_value

        if send_value != 0:
            send_transaction_amount += send_value
            if send_value > max_send_amount:
                max_send_amount = send_value
            if send_value < min_send_amount:
                min_send_amount = send_value

        receive_value = 0
        for one_output in outputs:
            if one_output["address"] == one_addr:
                receive_transaction_count += 1
                one_receive_value = float(one_output["value"])
                receive_value += one_receive_value

        if receive_value != 0:
            receive_transaction_amount += receive_value
            if receive_value > max_receive_amount:
                max_receive_amount = receive_value
            if receive_value < min_receive_amount:
                min_receive_amount = receive_value

    if receive_transaction_count != 0:
        avg_receive_amount = receive_transaction_amount / receive_transaction_count
    if send_transaction_count != 0:
        avg_send_amount = send_transaction_amount / send_transaction_count


    start_active_time = min(transaction_time_list)
    end_active_time = max(transaction_time_list)

        # logger.info(str(p_id) + ":error:" + one_addr)

    active_period = end_active_time - start_active_time
    return [one_addr, transaction_count, receive_transaction_count, receive_transaction_amount,
         avg_receive_amount, max_receive_amount, min_receive_amount, send_transaction_count,
         send_transaction_amount, avg_send_amount, max_send_amount, min_send_amount, max_inputs_count,
         min_inputs_count, max_outputs_count, min_outputs_count, has_mixing, has_locktime, active_period]

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

def already_extract_feature_address():
    already_address_set = set()
    year_random_white_address_path = "../nfs/wk/TDSC/random_white_address/"
    year_file_list = os.listdir(year_random_white_address_path)
    for year_file in year_file_list:
        year_file_path = year_random_white_address_path + year_file
        year_file_f = open(year_file_path, "r")
        year_file_reader = csv.reader(year_file_f)
        for row in year_file_reader:
            already_address_set.add(row[0])

    return already_address_set


def process(one_addr):
    if one_addr in already_address_set:
        return

    old_feature = old_addr_feature_db.get(bytes(one_addr, encoding='utf-8'))
    if old_feature is None:
        old_feature = extract_white_address_features(one_addr)
        new_feature_save_writer.writerow(old_feature)
        return 

    old_feature = str(old_feature, encoding='utf-8')
    if "inf" in old_feature:
        old_feature = extract_white_address_features(one_addr)
        new_feature_save_writer.writerow(old_feature)
        return 

    old_feature = eval(old_feature)
    old_tx_num = int(old_feature[0])

    addr_related_txs = addr_txs_db.get(bytes(one_addr, encoding='utf-8'))
    addr_related_txs = str(addr_related_txs, encoding='utf-8')
    addr_related_txs = eval(addr_related_txs)
    new_tx_num = len(addr_related_txs)


    if old_tx_num == new_tx_num:
        # 判断是否参与混币交易
        has_mixing = 0
        for this_tx_hash in addr_related_txs:
            if this_tx_hash in mixing_transaction_set:
                has_mixing = 1
        old_feature.insert(0, one_addr)
        old_feature.insert(16, has_mixing)
    
    else:
        old_feature = extract_white_address_features(one_addr)
    
    new_feature_save_writer.writerow(old_feature)


if __name__ == '__main__':

    # new_feature_save_path = "../addr_oppos_feature/white/year_random_addr_oppos_feature04.csv"
    # new_feature_save_f = open(new_feature_save_path, "a+")
    # new_feature_save_writer = csv.writer(new_feature_save_f)

    mixing_transaction_set = get_mixing_transactions()
    logger.info("mixing_transaction_set finish")

    already_address_set = already_extract_feature_address()
    logger.info("already_address_set already")

    # not_finish_oppos_path = "../nfs/wk/TDSC/white_address_oppos/not_finish_white_oppos.csv"
    # not_finish_oppos_f = open(not_finish_oppos_path, "r")
    # not_finish_oppos_reader = csv.reader(not_finish_oppos_f)

    # # long_address_list = ["1dice8EMZmqKvrGE4Qc9bUFf9PX3xaYDp", "1LuckyR1fFHEsXYyx5QK4UFzv3PEAepPMK", "3CD1QW6fjgTwKq3Pj97nty28WZAVkziNom", "1dice97ECuByXAvqXpaYzSaQuPVvrtmz6", "1G47mSr3oANXMafVrR8UC4pzV7FEAzo3r9", "1NDyJtNTjmwk5xPNhjgAMu4HDHigtobu1s"]
    # for addr in not_finish_oppos_reader:
    #     addr = addr[0]
    #     logger.info(addr)
    #     feature = extract_white_address_features(addr)
    #     new_feature_save_writer.writerow(feature)






    oppos_address_path = "../nfs/wk/TDSC/ransom_address_oppos/all_oppos.csv"
    oppos_address_f = open(oppos_address_path, "r")
    oppos_address_reader = csv.reader(oppos_address_f)

    count = 0
    po = Pool(10)
    for row in oppos_address_reader:
        one_addr = row[0]
        po.apply_async(process, (one_addr,))

    po.close()
    po.join()




# over_length: 1dice8EMZmqKvrGE4Qc9bUFf9PX3xaYDp, 1LuckyR1fFHEsXYyx5QK4UFzv3PEAepPMK, 3CD1QW6fjgTwKq3Pj97nty28WZAVkziNom, 1dice97ECuByXAvqXpaYzSaQuPVvrtmz6, 1G47mSr3oANXMafVrR8UC4pzV7FEAzo3r9, 1NDyJtNTjmwk5xPNhjgAMu4HDHigtobu1s



        # if one_addr in already_address_set:
        #     continue

        # old_feature = old_addr_feature_db.get(bytes(one_addr, encoding='utf-8'))
        # if old_feature is None:
        #     old_feature = extract_white_address_features(one_addr)
        #     new_feature_save_writer.writerow(old_feature)
        #     continue

        # old_feature = str(old_feature, encoding='utf-8')
        # if "inf" in old_feature:
        #     old_feature = extract_white_address_features(one_addr)
        #     new_feature_save_writer.writerow(old_feature)
        #     continue

        # old_feature = eval(old_feature)
        # old_tx_num = int(old_feature[0])

        # addr_related_txs = addr_txs_db.get(bytes(one_addr, encoding='utf-8'))
        # addr_related_txs = str(addr_related_txs, encoding='utf-8')
        # addr_related_txs = eval(addr_related_txs)
        # new_tx_num = len(addr_related_txs)


        # if old_tx_num == new_tx_num:
        #     # 判断是否参与混币交易
        #     has_mixing = 0
        #     for this_tx_hash in addr_related_txs:
        #         if this_tx_hash in mixing_transaction_set:
        #             has_mixing = 1
        #     old_feature.insert(0, one_addr)
        #     old_feature.insert(16, has_mixing)
        
        # else:
        #     old_feature = extract_white_address_features(one_addr)
        
        # new_feature_save_writer.writerow(old_feature)
        

