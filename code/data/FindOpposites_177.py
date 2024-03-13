#coding=UTF-8
import csv
import json
import multiprocessing
import os
import random
import sys
import logging
from multiprocessing import Process
import threading

from loguru import logger
import rocksdb


def find_op_addrs(sample_addr):
    in_neighbor = []
    out_neighbor = []
    out_sibling = []
    in_sibling = []
    # dict结构：{'addr':[in_neighbor, outneighbor, in_sib, out_sib]}
    addr_info_dict = {}

    addr_related_txs = db.get(bytes(sample_addr, encoding='utf-8'))
    if addr_related_txs is not None:
        addr_related_txs = str(addr_related_txs, encoding='utf-8')
        addr_related_txs = fix_data_format(addr_related_txs)
        addr_related_txs = eval(addr_related_txs)
        for one_tx in addr_related_txs:
            one_tx = one_tx.split('/')
            this_tx_hash = one_tx[-1]
            tx_detail_info = db_tx.get(bytes(this_tx_hash, encoding='utf-8'))
            if(tx_detail_info is None):
                continue
            tx_detail_info_json = json.loads(str(tx_detail_info, encoding='utf-8'))
            vin=tx_detail_info_json["vin"]
            vout=tx_detail_info_json["vout"]
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
    else:
        logger.error(sample_addr)
    try:
     addr_info_dict.pop(sample_addr)
    except Exception as ex:
        pass

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


def find_op_addrs_only_opposite(sample_addr):
    addr_related_txs = db.get(bytes(sample_addr, encoding='utf-8'))

    try:
        addr_related_txs = str(addr_related_txs, encoding='utf-8')
        addr_related_txs = fix_data_format(addr_related_txs)
        addr_related_txs = eval(addr_related_txs)
    except:
        logger.info(sample_addr + ' error')
        return 'addr_error'

    sample_opposite_addrs = set()
    tmp_vin = set()
    tmp_vout = set()

    for one_tx in addr_related_txs:
        try:
            one_tx = one_tx.split('/')
        except:
            print(one_tx)
            print(type(one_tx))
            exit()
        tx_date = one_tx[0]
        tx_block = one_tx[1]
        tx_hash = one_tx[2]
        tx_loca = '../cd_data/bitcoin_data/' + tx_date + '/' + tx_block + '/' + tx_hash + '.json'
        for line in open(tx_loca, 'r'):
            if 'coinbase' in line:
                line = json.loads(line)
                vin = None
                vout = line['vout']
            else:
                line = json.loads(line)
                vin = line['vin']
                vout = line['vout']
            if vin is not None:
                for vin_item in vin:
                    tmp_op_addr = vin_item.get('address')
                    if tmp_op_addr is not None and tmp_op_addr is sample_addr:
                        tmp_vin.add(tmp_op_addr)
            for vout_item in vout:
                tmp_op_addr = vout_item.get('address')
                if tmp_op_addr is not None:
                    tmp_vout.add(tmp_op_addr)

        if sample_addr in tmp_vin:
            sample_opposite_addrs = sample_opposite_addrs.union(tmp_vout)

        if sample_addr in tmp_vout:
            sample_opposite_addrs = sample_opposite_addrs.union(tmp_vin)

        tmp_vin.clear()
        tmp_vout.clear()

    sample_opposite_addrs.discard(sample_addr)
    return sample_opposite_addrs


def sample_addr_feature_cal(sample_addr, addr_related_txs):
    # addr_related_txs = db.get(bytes(sample_addr, encoding='utf-8'))
    # try:
    #     addr_related_txs = str(addr_related_txs, encoding='utf-8')
    #     addr_related_txs = fix_data_format(addr_related_txs)
    #     addr_related_txs = eval(addr_related_txs)
    #     # db.put(bytes(sample_addr, encoding='utf-8'), bytes(str(addr_related_txs), encoding='utf-8'))
    # except:
    #     logger.info(sample_addr + ' error')
    #     return None

    transaction_dict_list = []
    for one_tx in addr_related_txs:

        try:
            one_tx = one_tx.split('/')
        except:
            logger.info(sample_addr)
            logger.info(one_tx)
            logger.info(addr_related_txs)
            logger.info(type(one_tx))
            logger.info(type(addr_related_txs))
            exit()
        tx_date = one_tx[0]
        tx_block = one_tx[1]
        tx_hash = one_tx[2]
        tx_loca = '../cd_data/bitcoin_data/' + tx_date + '/' + tx_block + '/' + tx_hash + '.json'
        for line in open(tx_loca, 'r'):
            line = json.loads(line)
            transaction_dict_list.append(line)
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
            if one_input.get("address") == sample_addr:
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
            if one_output["address"] == sample_addr:
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

    active_period = end_active_time - start_active_time

    addr_feature = [sample_addr, transaction_count, receive_transaction_count, receive_transaction_amount,
                    avg_receive_amount,
                    max_receive_amount, min_receive_amount, send_transaction_count, send_transaction_amount,
                    avg_send_amount, max_send_amount,
                    min_send_amount, max_inputs_count,
                    min_inputs_count,
                    max_outputs_count, min_outputs_count,
                    has_locktime, active_period]
    return addr_feature


def opposite_addr_feature(oppo_addr):
    addr_related_txs = db.get(bytes(oppo_addr, encoding='utf-8'))
    try:
        addr_related_txs = str(addr_related_txs, encoding='utf-8')
        addr_related_txs = fix_data_format(addr_related_txs)
        addr_related_txs = eval(addr_related_txs)
        # db.put(bytes(sample_addr, encoding='utf-8'), bytes(str(addr_related_txs), encoding='utf-8'))
    except:
        logger.info(oppo_addr + ' error')
        return None

    transaction_dict_list = []
    for one_tx in addr_related_txs:

        try:
            one_tx = one_tx.split('/')
        except:
            logger.info(oppo_addr)
            logger.info(one_tx)
            logger.info(addr_related_txs)
            logger.info(type(one_tx))
            logger.info(type(addr_related_txs))
            exit()
        tx_date = one_tx[0]
        tx_block = one_tx[1]
        tx_hash = one_tx[2]
        tx_loca = '../cd_data/bitcoin_data/' + tx_date + '/' + tx_block + '/' + tx_hash + '.json'
        for line in open(tx_loca, 'r'):
            line = json.loads(line)
            transaction_dict_list.append(line)
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
            if one_input.get("address") == oppo_addr:
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
            if one_output["address"] == oppo_addr:
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

    active_period = end_active_time - start_active_time

    addr_feature = [oppo_addr, transaction_count, receive_transaction_count, receive_transaction_amount,
                    avg_receive_amount,
                    max_receive_amount, min_receive_amount, send_transaction_count, send_transaction_amount,
                    avg_send_amount, max_send_amount,
                    min_send_amount, max_inputs_count,
                    min_inputs_count,
                    max_outputs_count, min_outputs_count,
                    has_locktime, active_period]
    return addr_feature


def fix_data_format(data):
    data = data.replace('[', '')
    data = data.replace(']', '')
    data = '[' + data + ']'
    return data


# 切分list
def func(listTemp, n):
    for i in range(0, len(listTemp), n):
        yield listTemp[i:i + n]


# def process_multi(opposite_addr, p_id):
#     write_list = {}
#     percent = len(opposite_addr)
#     with open('../wk/data_new_2/random_rocksdb_file_2w_20w/' + str(p_id) + '_data.csv', 'a') as write_file:
#         for i in range(percent):
#             one_oppo_addr = opposite_addr[i]
#             if one_oppo_addr is '0000000000000000000000000000000000':
#                 continue
#             check = db3.get(bytes(one_oppo_addr, encoding='utf-8'))
#             if check is not None:
#                 continue
#             f = opposite_addr_feature(one_oppo_addr)
#             if f is None:
#                 continue
#             else:
#                 write_file.write(one_oppo_addr + ';' + str(f) + '\n')
#
#             if i % 100 == 0:
#                 logger.info(str(p_id) + '---===/// ' + str(i) + ' %%% ' + str(percent))
#         write_list[one_oppo_addr] = f
#         if i % 20 == 0:
#             for item in write_list:
#                 db3.put(bytes(item, encoding='utf-8'), bytes(str(write_list[item]), encoding='utf-8'))
#             write_list.clear()
#     logger.info(str(p_id) + ':...___+++===' + str(i) + '.../' + str(percent))
# for item in write_list:
#     db3.put(bytes(item, encoding='utf-8'), bytes(str(write_list[item]), encoding='utf-8'))


def test():
    pass


# class myThread(threading.Thread):
#     def __init__(self, threadID, addr_list):
#         threading.Thread.__init__(self)
#         self.threadID = threadID
#         self.addr_list = addr_list
#
#     def run(self):
#         print("开始线程：" + self.name)
#         process_multi(self.addr_list, self.threadID)
#         print("退出线程：" + self.name)

def random_samples(range, nums):
    random_nums_set = set()
    while len(random_nums_set) < nums:
        tmp_random_num = random.randint(0, range)
        random_nums_set.add(tmp_random_num)
    logger.info(len(random_nums_set))
    return random_nums_set


def process_multi_oppo(rs):
    cnt = 0
    file_name = rs[0:-21]
    os.mkdir('../wk/data_new_2/paper_dataset_new_oppo_addrs/' + file_name)
    for line in open('../wk/TWEB/ransomware_address_codes/paper_dataset_feature/' + rs):
        line = line.split(',')
        addr = line[0]
        res = find_op_addrs(addr)
        if res != 'addr_error':
            addr_in_neighbor = res[0]
            addr_outneighbor = res[1]
            addr_in_sib = res[2]
            addr_out_sib = res[3]

            with open('../wk/data_new_2/2016/' + file_name + '/in_neighbor.csv',
                      'a') as f1:
                    f1.write(addr+";"+str(addr_in_neighbor) + '\n')
            with open('../wk/data_new_2/2016/' + file_name + '/out_neighbor.csv',
                      'a') as f2:
                    f2.write(addr+";"+str(addr_outneighbor)  + '\n')
            with open('../wk/data_new_2/2016/' + file_name + '/in_sib.csv',
                      'a') as f3:
                    f3.write(addr+";"+str(addr_in_sib)  + '\n')
            with open('../wk/data_new_2/2016/' + file_name + '/out_sib.csv',
                      'a') as f4:
                    f4.write(addr+";"+str(addr_out_sib)  + '\n')
        cnt += 1
        logger.info(file_name + ' find... ' + str(cnt))
    logger.info(file_name + ' finished...')


def process_multi_oppo_year(year):
    cnt = 0
    if os.path.exists('../nfs/tmk/www_experiment/result_csv/2021/' + year) is False:
        os.makedirs('../nfs/tmk/www_experiment/result_csv/2021/' + year)
    f1 = open('../nfs/tmk/www_experiment/result_csv/2021/' + year + '/in_neighbor.csv','a')
    f2 = open('../nfs/tmk/www_experiment/result_csv/2021/' + year + '/out_neighbor.csv','a')
    f3 = open('../nfs/tmk/www_experiment/result_csv/2021/' + year + '/in_sib.csv','a')
    f4 = open('../nfs/tmk/www_experiment/result_csv/2021/' + year + '/out_sib.csv','a')
    for line in open('../nfs/tmk/www_experiment/resource_csv/2021/' + year + '.csv',
                     'r'):
        if '\n' in line:
            addr = line[0:-1]
        else:
            addr = line

        res = find_op_addrs(addr)

        if res != 'addr_error':
            addr_in_neighbor = res[0]
            addr_outneighbor = res[1]
            addr_in_sib = res[2]
            addr_out_sib = res[3]

            f1.write(addr+";"+str(addr_in_neighbor) + '\n')
            f2.write(addr+";"+str(addr_outneighbor) + '\n')
            f3.write(addr+";"+str(addr_in_sib) + '\n')
            f4.write(addr+";"+str(addr_out_sib) + '\n')
        cnt += 1
        if cnt % 1000 == 0:
            logger.info(year + ' find... ' + str(cnt))

    logger.info(year + ' finished...')


if __name__ == '__main__':
    db = rocksdb.DB('../addr_related_txs.db', rocksdb.Options(create_if_missing=True), read_only=True)
    db_tx = rocksdb.DB('../transaction_new', rocksdb.Options(create_if_missing=True), read_only=True)
    # all_rs_family = os.listdir('../wk/TWEB/ransomware_address_codes/paper_dataset_feature/')
    # all_rs_family.sort()
 #   years = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
    years=["0","1","2","3","4","5","6","7","8","9"]

    process_list = []
    for y in years:
        process_X = Process(target=process_multi_oppo_year, args=(str(y),))
        process_list.append(process_X)

    for p in process_list:
        p.daemon = True
        p.start()

    for p in process_list:
        p.join()


    # db2 = rocksdb.DB('../wk/data_new_2/sample_addr_opposite_addrs.db',
    #                  rocksdb.Options(create_if_missing=True))
    # db3 = rocksdb.DB('../wk/data_new_2/2w_sample_addr_feature.db', rocksdb.Options(create_if_missing=True))
    # samples = []
    # process_list = []
    # sample_op_addr_tmp = {}
    # all_oppo_addrs_set = set()
    # all_oppo_addrs_list = []
    # for line in open(random_samples_loca, 'r'):
    #     if '\n' in line:
    #         one_sample_addr = line[0:-1]
    #     else:
    #         one_sample_addr = line
    #     if one_sample_addr is None:
    #         logger.info(one_sample_addr)
    #         exit()
    #     samples.append(one_sample_addr)
    # logger.info('start...')
    # with open('../wk/data_new/random_nums.txt', 'a') as f:
    #     f.write(str(list(random_nums)))
    # for one_sample_addr in samples:
    #     cnt += 1
    #     if cnt not in random_nums:
    #         continue
    #     elif cnt in random_nums:
    #         res_list = db2.get(bytes(one_sample_addr, encoding='utf-8'))
    #         if res_list is None:
    #             continue
    #         else:
    #             res_list = eval(str(res_list, encoding='utf-8'))
    #             all_oppo_addrs_list.extend(res_list)
    #         if cnt % 1000 == 0:
    #             all_oppo_addrs_list = list(set(all_oppo_addrs_list))
    #             logger.info(str(cnt) + ' ...processed')
    #     else:
    #         break
    # all_oppo_addrs_set = list(set(all_oppo_addrs_list))

    # # 样本地址的对手地址
    # res_list = find_op_addrs_only_opposite(one_sample_addr)
    # if res_list is 'addr_error':
    #     logger.info(one_sample_addr + ' sssdddd')
    #     continue
    # elif len(res_list) > 0:
    #     res = list(res_list)
    #     sample_op_addr_tmp[one_sample_addr] = res
    #     all_oppo_addrs_list.extend(res)
    # if cnt % 1000 == 0:
    #     for item in sample_op_addr_tmp:
    #         db2.put(bytes(item, encoding='utf-8'), bytes(str(sample_op_addr_tmp[item]), encoding='utf-8'))
    #     all_oppo_addrs_list = list(set(all_oppo_addrs_list))

    # step_len = int(len(all_oppo_addrs_set) / 15)
    # logger.info(step_len)
    # del db2
    # for j in range(15):
    #     if j == 14:
    #         tmp_sample = all_oppo_addrs_set[j * step_len:]
    #     else:
    #         tmp_sample = all_oppo_addrs_set[j * step_len:j * step_len + step_len]
    #     process_X = Process(target=process_multi,
    #                         args=(tmp_sample, j,))
    #     # thread_x = myThread(j, tmp_sample)
    #     process_list.append(process_X)
    #
    # for p in process_list:
    #     p.daemon = True
    #     p.start()
    #
    # for p in process_list:
    #     p.join()

    # check_exist_oppo_info = db2.get(bytes(one_sample_addr, encoding='utf-8'))
    # if check_exist_oppo_info is not None:
    #     continue

    # 样本地址的对手地址
    # res_list = find_op_addrs(one_sample_addr)
    # if res_list is 'addr_error':
    #     logger.info(one_sample_addr + ' sssdddd')
    #     continue
    # elif len(res_list[0]) > 0:
    #     res = list(res_list[0])
    #     db2.put(bytes(one_sample_addr, encoding='utf-8'), bytes(str(res), encoding='utf-8'))

    # 样本地址feature
    # check_exist_sample = db3.get(bytes(one_sample_addr, encoding='utf-8'))
    # if check_exist_sample is None:
    #     res2 = sample_addr_feature_cal(one_sample_addr, res_list[1])
    #     if res2 is not None:
    #         db3.put(bytes(one_sample_addr, encoding='utf-8'), bytes(str(res2), encoding='utf-8'))

    # 对手地址feature
    # for one_op_addr in res_list[0]:
    #     check_exist = db3.get(bytes(one_op_addr, encoding='utf-8'))
    #     if check_exist is None:
    #         op_addr_feature = opposite_addr_feature(one_op_addr)
    #         if op_addr_feature is not None:
    #             db3.put(bytes(one_op_addr, encoding='utf-8'), bytes(str(op_addr_feature), encoding='utf-8'))
#
