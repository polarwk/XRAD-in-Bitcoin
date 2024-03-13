import csv
import json
import multiprocessing
import os
import random
import sys
import logging
from multiprocessing import Process
import threading
csv.field_size_limit(1024*1024*1024)

from loguru import logger
import rocksdb


addr_txs_path = "../cd_data/tsdc/addr_related_txs.db"
addr_txs_db = rocksdb.DB(addr_txs_path, rocksdb.Options(create_if_missing=True))
tx_detail_path = "../cd_data/bitcoin_transaction.db"
tx_detail_db = rocksdb.DB(tx_detail_path, rocksdb.Options(create_if_missing=True))

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



def find_op_addrs(sample_addr):
    in_neighbor = []
    out_neighbor = []
    out_sibling = []
    in_sibling = []
    # dict结构：{'addr':[in_neighbor, out_neighbor, in_sib, out_sib]}
    addr_info_dict = {}

    addr_related_txs = addr_txs_db.get(bytes(sample_addr, encoding='utf-8'))
    if addr_related_txs is not None:
        addr_related_txs = str(addr_related_txs, encoding='utf-8')
        addr_related_txs = eval(addr_related_txs)
        for this_tx_hash in addr_related_txs:
            # 过滤混币交易
            if this_tx_hash in mixing_transaction_set:
                # logger.info(sample_addr + ":mixing")
                continue
            tx_detail_info = tx_detail_db.get(bytes(this_tx_hash, encoding='utf-8'))
            if tx_detail_info is None:
                logger.info(sample_addr)
                logger.info(this_tx_hash)
            tx_detail_info_json = json.loads(str(tx_detail_info, encoding='utf-8'))
            vin = tx_detail_info_json["vin"]
            vout = tx_detail_info_json["vout"]
            if vin is not None:
                if (sample_addr in str(vout)) and (sample_addr not in str(vin)):
                    for vin_item in vin:
                        tmp_op_addr = vin_item.get('address')
                        if (tmp_op_addr is not None) and (tmp_op_addr != '0000000000000000000000000000000000'):
                            if tmp_op_addr not in addr_info_dict:
                                addr_info_dict[tmp_op_addr] = [1, 0, 0, 0]
                            else:
                                tmp_val = addr_info_dict[tmp_op_addr]
                                tmp_val[0] = tmp_val[0] + 1
                                addr_info_dict[tmp_op_addr] = tmp_val

                    for vout_item in vout:
                        tmp_op_addr = vout_item.get('address')
                        if (tmp_op_addr is not None) and (tmp_op_addr != '0000000000000000000000000000000000'):
                            if tmp_op_addr not in addr_info_dict:
                                addr_info_dict[tmp_op_addr] = [0, 0, 0, 1]
                            else:
                                tmp_val = addr_info_dict[tmp_op_addr]
                                tmp_val[3] = tmp_val[3] + 1
                                addr_info_dict[tmp_op_addr] = tmp_val
                else:
                    for vin_item in vin:
                        tmp_op_addr = vin_item.get('address')
                        if (tmp_op_addr is not None) and (tmp_op_addr != '0000000000000000000000000000000000'):
                            if tmp_op_addr not in addr_info_dict:
                                addr_info_dict[tmp_op_addr] = [0, 0, 1, 0]
                            else:
                                tmp_val = addr_info_dict[tmp_op_addr]
                                tmp_val[2] = tmp_val[2] + 1
                                addr_info_dict[tmp_op_addr] = tmp_val

                    for vout_item in vout:
                        tmp_op_addr = vout_item.get('address')
                        if (tmp_op_addr is not None) and (tmp_op_addr != '0000000000000000000000000000000000'):
                            if tmp_op_addr not in addr_info_dict:
                                addr_info_dict[tmp_op_addr] = [0, 1, 0, 0]
                            else:
                                tmp_val = addr_info_dict[tmp_op_addr]
                                tmp_val[1] = tmp_val[1] + 1
                                addr_info_dict[tmp_op_addr] = tmp_val
            else:
                for vout_item in vout:
                    tmp_op_addr = vout_item.get('address')
                    if (tmp_op_addr is not None) and (tmp_op_addr != '0000000000000000000000000000000000'):
                        if tmp_op_addr not in addr_info_dict:
                            addr_info_dict[tmp_op_addr] = [0, 0, 0, 1]
                        else:
                            tmp_val = addr_info_dict[tmp_op_addr]
                            tmp_val[3] = tmp_val[3] + 1
                            addr_info_dict[tmp_op_addr] = tmp_val

    if sample_addr in addr_info_dict:
        addr_info_dict.pop(sample_addr)

    for one_addr in addr_info_dict:
        main_role_val = max(addr_info_dict[one_addr])
        role = addr_info_dict[one_addr].index(main_role_val)
        if role == 0:
            in_neighbor.append(one_addr)
        elif role == 1:
            out_neighbor.append(one_addr)
        elif role == 2:
            out_sibling.append(one_addr)
        elif role == 3:
            in_sibling.append(one_addr)
    in_neighbor = list(set(in_neighbor))
    out_neighbor = list(set(out_neighbor))
    out_sibling = list(set(out_sibling))
    in_sibling = list(set(in_sibling))

    return [in_neighbor, out_neighbor, out_sibling, in_sibling]


def process_year_white_address(addr_list, pid):

    root_path = "../nfs/wk/TDSC/white_address_oppos/" + str(pid) + "/"
    if os.path.exists(root_path) is False:
        os.makedirs(root_path)

    f1 = open(root_path + 'in_neighbor.csv', 'a+')
    f2 = open(root_path + 'out_neighbor.csv', 'a+')
    f3 = open(root_path + 'in_sib.csv', 'a+')
    f4 = open(root_path + 'out_sib.csv', 'a+')

    f1_writer = csv.writer(f1, delimiter=";", quoting=csv.QUOTE_NONE, quotechar='')
    f2_writer = csv.writer(f2, delimiter=";", quoting=csv.QUOTE_NONE, quotechar='')
    f3_writer = csv.writer(f3, delimiter=";", quoting=csv.QUOTE_NONE, quotechar='')
    f4_writer = csv.writer(f4, delimiter=";", quoting=csv.QUOTE_NONE, quotechar='')
    cnt = 0
    length = len(addr_list)
    for addr in addr_list:
        if cnt % 1000 == 0:
            logger.info(str(pid) + "       :       " + str(cnt) + "//" + str(length))

        cnt += 1

        res = find_op_addrs(addr)

        addr_in_neighbor = res[0]
        addr_out_neighbor = res[1]
        addr_in_sib = res[2]
        addr_out_sib = res[3]

        f1_writer.writerow([addr, str(addr_in_neighbor)])
        f2_writer.writerow([addr, str(addr_out_neighbor)])
        f3_writer.writerow([addr, str(addr_in_sib)])
        f4_writer.writerow([addr, str(addr_out_sib)])

    logger.info(str(pid) + "           finish")


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


def process_oppo_address():
    logger.info("process_oppo_address start")
    oppo_address_list = []

    oppos_path = "../nfs/wk/TDSC/white_address_oppos/"
    for year in range(2013, 2021):
        logger.info(year)
        year_oppos_path = oppos_path + str(year) + "/"
        oppo_role_list = os.listdir(year_oppos_path)
        for oppo_role in oppo_role_list:
            oppo_role_path = year_oppos_path + oppo_role
            oppo_role_f = open(oppo_role_path, "r")
            oppo_role_reader = csv.reader(oppo_role_f, delimiter=";", quoting=csv.QUOTE_NONE, quotechar='')
            for row in oppo_role_reader:
                for addr in eval(row[1]):
                    oppo_address_list.append(addr)
    oppo_address_set = set(oppo_address_list)
    save_path = "../nfs/wk/TDSC/white_address_oppos/year_random_white_oppos.csv"
    save_f = open(save_path, "a+")
    save_writer = csv.writer(save_f)
    for addr in oppo_address_set:
        save_writer.writerow([addr])




if __name__ == '__main__':
    process_oppo_address()
    # mixing_transaction_set = get_mixing_transactions()
    #
    # process_list = []
    #
    # for i in range(2013, 2021):
    #     white_address_list = []
    #     white_address_path = "../nfs/wk/TDSC/random_white_address/white_" + str(i) + ".csv"
    #     white_address_f = open(white_address_path, "r")
    #     white_address_reader = csv.reader(white_address_f)
    #     for row in white_address_reader:
    #         white_address_list.append(row[0])
    #     process_i = Process(target=process_year_white_address,
    #                         args=(white_address_list, i,))
    #     process_list.append(process_i)
    #
    # for p in process_list:
    #     p.start()
    #
    # for p in process_list:
    #     p.join()