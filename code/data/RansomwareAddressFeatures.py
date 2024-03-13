import os, sys
from loguru import logger
import json
import csv
from multiprocessing import Process
import rocksdb

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("RansomwareAddressFeatures")


def fix_data_format(data):
    data = data.replace('[', '')
    data = data.replace(']', '')
    data = '[' + data + ']'
    return data


def find_related_address_of_an_address():
    path = "../wk/data_new_2/zhaoyu_result/"
    related_address_tag_path = "../wk/TWEB/ransomware_address_codes/AddressFeatures/related_address_tag.csv"
    csv_f = open(related_address_tag_path, "a+")
    csv_writer = csv.writer(csv_f)
    tag_list = os.listdir(path)
    for tag in tag_list:
        if tag == "Unknown":
            continue
        tag_path = path + tag + "/"
        address_file_list = os.listdir(tag_path)
        related_address_set = set()
        for address_file in address_file_list:
            address = address_file.split(".json")[0]
            with open(tag_path + address_file) as f:
                for row in f:
                    transaction = json.loads(row)
                    inputs = transaction["vin"]
                    outputs = transaction["vout"]
                    for one_input in inputs:
                        address = one_input.get("address")
                        if address is not None:
                            related_address_set.add(one_input["address"])
                    for one_output in outputs:
                        address = one_output.get("address")
                        if address is not None:
                            related_address_set.add(one_output["address"])

        for address in related_address_set:
            csv_writer.writerow([address, tag])


def extract_ransomware_related_address_features():
    ransomware_related_address_txs_path = "../cd_data/old_neighbor/"
    ransomware_related_address_features_path = "../wk/TWEB/ransomware_address_codes/AddressFeatures/ransomware_related_address_features/"
    # features_save_path = "../wk/TWEB/ransomware_address_codes/AddressFeatures/ransom_address_feature.csv"
    # f = open(features_save_path, "a+")
    # csv_writer = csv.writer(f)
    # csv_writer.writerow(
    #     ["address", "tag", "transaction_count", "receive_transaction_count", "receive_transaction_amount", "avg_receive_amount",
    #      "max_receive_amount", "min_receive_amount", "send_transaction_count", "send_transaction_amount",
    #      "avg_send_amount", "max_send_amount", "min_send_amount", "max_inputs_count", "min_inputs_count",
    #      "max_outputs_count", "min_outputs_count", "has_locktime", "active_period"])

    tag_list = os.listdir(ransomware_related_address_txs_path)
    # tag_list = ["APT", "Avaddon", "BadRabbit", "Bucbi", "Cerber", "CryptConsole", "CryptoDefense", "Cryptohitman", "CryptoHost", "CryptoLocker", "CryptoTorLocker2015", "CryptoWall", "CryptXXX", "Locky", "WannaCry"]
    for tag in tag_list:
        if tag == "Unknown":
            continue
        logger.info(tag)
        # 地址交易数据
        path = ransomware_related_address_txs_path + tag + "/"

        # 地址feature路径
        tag_path = ransomware_related_address_features_path + tag + "/"
        if not os.path.exists(tag_path):
            os.mkdir(tag_path)

        address_file_list = os.listdir(path)
        for address_file in address_file_list:
            transaction_dict_list = []
            address = address_file.split(".json")[0]
            logger.info(address)
            if address == "0000000000000000000000000000000000":
                continue
            address_feature_path = tag_path + address + ".csv"
            feature_f = open(address_feature_path, "a+")
            feature_writer = csv.writer(feature_f)
            with open(path + address_file, "r") as f:
                for row in f:
                    transaction = json.loads(row)
                    transaction_dict_list.append(transaction)
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
                    if one_input.get("address") == address:
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
                    if one_output["address"] == address:
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

            feature_writer.writerow(
                [address, tag, transaction_count, receive_transaction_count, receive_transaction_amount,
                 avg_receive_amount, max_receive_amount, min_receive_amount, send_transaction_count,
                 send_transaction_amount, avg_send_amount, max_send_amount, min_send_amount, max_inputs_count,
                 min_inputs_count, max_outputs_count, min_outputs_count, has_locktime, active_period])


def ransom_address_features():
    ransomware_address_txs_path = "../wk/data_new_2/zhaoyu_result/"
    features_save_path = "../wk/TWEB/ransomware_address_codes/AddressFeatures/ransom_address_features.csv"

    f = open(features_save_path, "a+")
    csv_writer = csv.writer(f)
    csv_writer.writerow(
        ["address", "tag", "transaction_count", "receive_transaction_count", "receive_transaction_amount",
         "avg_receive_amount",
         "max_receive_amount", "min_receive_amount", "send_transaction_count", "send_transaction_amount",
         "avg_send_amount", "max_send_amount", "min_send_amount", "max_inputs_count", "min_inputs_count",
         "max_outputs_count", "min_outputs_count", "has_locktime", "active_period"])

    tag_list = os.listdir(ransomware_address_txs_path)
    for tag in tag_list:
        tag_path = ransomware_address_txs_path + tag + "/"

        if tag == "Unknown":
            continue
        logger.info(tag)
        # path = ransomware_address_txs_path + tag + "/"

        address_file_list = os.listdir(tag_path)
        write_features_list = []
        count = 0
        for address_file in address_file_list:
            logger.info(count)
            count += 1
            transaction_dict_list = []
            address = address_file.split(".json")[0]
            # logger.info(address)
            with open(tag_path + address_file, "r") as f:
                for row in f:
                    transaction = json.loads(row)
                    transaction_dict_list.append(transaction)
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
                    if one_input.get("address") == address:
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
                    if one_output["address"] == address:
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
            # write_features_list.append(
            #     [address, tag, transaction_count, receive_transaction_count, receive_transaction_amount,
            #      avg_receive_amount, max_receive_amount, min_receive_amount, send_transaction_count,
            #      send_transaction_amount, avg_send_amount, max_send_amount, min_send_amount, max_inputs_count,
            #      min_inputs_count, max_outputs_count, min_outputs_count, has_locktime, active_period])

            csv_writer.writerow([address, tag, transaction_count, receive_transaction_count, receive_transaction_amount,
                                 avg_receive_amount, max_receive_amount, min_receive_amount, send_transaction_count,
                                 send_transaction_amount, avg_send_amount, max_send_amount, min_send_amount,
                                 max_inputs_count, min_inputs_count, max_outputs_count, min_outputs_count, has_locktime,
                                 active_period])
        # csv_writer.writerows(write_features_list)
        # f.close()


def extract_ransomware_address_features():
    ransomware_address_txs_path = "../nfs/RansomwareAddressDetection/combined_txs/"
    features_save_path = "../nfs/RansomwareAddressDetection/related_address_features/"

    tag_list = os.listdir(ransomware_address_txs_path)
    for tag in tag_list:
        tag_path = ransomware_address_txs_path + tag + "/"

        f = open(features_save_path + tag + ".csv", "a+")
        csv_writer = csv.writer(f)
        csv_writer.writerow(
            ["address", "tag", "transaction_count", "receive_transaction_count", "receive_transaction_amount",
             "avg_receive_amount",
             "max_receive_amount", "min_receive_amount", "send_transaction_count", "send_transaction_amount",
             "avg_send_amount", "max_send_amount", "min_send_amount", "max_inputs_count", "min_inputs_count",
             "max_outputs_count", "min_outputs_count", "has_locktime", "active_period"])

        # tag_list = os.listdir(ransomware_address_txs_path)
        # tag_list = ["APT", "Avaddon", "BadRabbit", "Bucbi", "Cerber", "CryptConsole", "CryptoDefense", "Cryptohitman", "CryptoHost", "CryptoLocker", "CryptoTorLocker2015", "CryptoWall", "CryptXXX", "Locky", "WannaCry"]
        # for tag in tag_list:
        if tag == "Unknown":
            continue
        logger.info(tag)
        # path = ransomware_address_txs_path + tag + "/"

        address_file_list = os.listdir(tag_path)
        write_features_list = []
        count = 0
        for address_file in address_file_list:
            logger.info(count)
            count += 1
            transaction_dict_list = []
            address = address_file.split(".json")[0]
            # logger.info(address)
            with open(tag_path + address_file, "r") as f:
                for row in f:
                    transaction = json.loads(row)
                    transaction_dict_list.append(transaction)
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
                    if one_input.get("address:") == address:
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
                    if one_output["address"] == address:
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
            write_features_list.append(
                [address, tag, transaction_count, receive_transaction_count, receive_transaction_amount,
                 avg_receive_amount, max_receive_amount, min_receive_amount, send_transaction_count,
                 send_transaction_amount, avg_send_amount, max_send_amount, min_send_amount, max_inputs_count,
                 min_inputs_count, max_outputs_count, min_outputs_count, has_locktime, active_period])

            # csv_writer.writerow([address, tag, transaction_count, receive_transaction_count, receive_transaction_amount, avg_receive_amount, max_receive_amount, min_receive_amount, send_transaction_count, send_transaction_amount, avg_send_amount, max_send_amount,  min_send_amount, max_inputs_count, min_inputs_count, max_outputs_count, min_outputs_count, has_locktime, active_period])
        csv_writer.writerows(write_features_list)
        f.close()


# def find_txs_of_related_addresses(year):
#     year = str(year)
#     logger.info(year)
#     year_path = txs_path + year + "/"
#     date_list = os.listdir(year_path)
#     date_list = sorted(date_list)
#     for date in date_list:
#         write_dict = dict()
#         logger.info(date)
#         date_path = year_path + date + "/"
#         blocks_list = os.listdir(date_path)
#         for block in blocks_list:
#             block_path = date_path + block
#             with open(block_path, "r") as f:
#                 for row in f:
#                     address_set = set()
#                     transaction = json.loads(row)
#
#                     inputs = transaction["vin"]
#                     outputs = transaction["vout"]
#
#                     new_inputs = []
#                     for vin in inputs:
#                         address = vin.get("address")
#                         value = vin.get("value")
#                         new_inputs.append({"address:": address, "value": value})
#                         if address is not None:
#                             address_set.add(address)
#                     transaction["vin"] = new_inputs
#
#                     new_outputs = []
#                     for vout in outputs:
#                         address = vout.get("address")
#                         value = vout.get("value")
#                         new_outputs.append({"address": address, "value": value})
#                         if address is not None:
#                             address_set.add(address)
#                     transaction["vout"] = new_outputs
#
#                     overlap_address_set = address_set.intersection(related_address_set)
#
#                     for address in overlap_address_set:
#                         tag_list = address_tag_dict[address]
#                         for tag in tag_list:
#                             write_path = related_txs_path + tag + "/" + year + "/" + address + ".json"
#                             if write_path not in write_dict:
#                                 write_dict[write_path] = []
#                             write_dict[write_path].append(json.dumps(transaction))
#
#         for write_path, row_list in write_dict.items():
#             with open(write_path, "a+") as f:
#                 for row in row_list:
#                     f.write(row + "\n")


def process_2016_addr_feature():
    logger.info("random opposite addr not in feature")

    tx_data_loca = '../cd_data/transaction_new'
    addr_related_tx = '../cd_data/addr_related_txs.db'
    db_tx = rocksdb.DB(tx_data_loca, rocksdb.Options(create_if_missing=True), read_only=True)
    db_addr_tx = rocksdb.DB(addr_related_tx, rocksdb.Options(create_if_missing=True), read_only=True)
    addr_path = '../nfs/tmk/www_experiment/resource_csv/2021/2021_addr.csv'
    addr_list = []
    for line in open(addr_path, 'r'):
        if '\n' in line:
            one_addr = line[0:-1]
        else:
            one_addr = line
        addr_list.append(one_addr)

    # 没用的时候可以删除，为了去除重复地址
    # addr_list = list(set(addr_list))
    # addr_set = set(addr_list)

    # opposite_2016_feature_path = "../cd_data/2016_feature_0630/"
    # file_name_list = os.listdir(opposite_2016_feature_path)
    # opposite_2016_addr_list = []
    # for file_name in file_name_list:
    #     csv_reader = csv.reader(open(opposite_2016_feature_path + file_name, "r"))
    #     next(csv_reader)
    #     for row in csv_reader:
    #         if row[0] == "address":
    #             continue
    #         opposite_2016_addr_list.append(row[0])
    # addr_set = addr_set - set(opposite_2016_addr_list)
    # logger.info(len(addr_set))

    # address_13_list = []
    # addr_path = "../wk/data_new/2013_addr.csv"
    # for line in open(addr_path, 'r'):
    #     if '\n' in line:
    #         one_addr = line[0:-1]
    #     else:
    #         one_addr = line
    #     address_13_list.append(one_addr)
    # addr_set = addr_set - set(address_13_list)
    # logger.info("2013 remove")
    # logger.info(len(addr_set))
    #
    # address_14_list = []
    # addr_path = "../wk/data_new/2014_addr.csv"
    # for line in open(addr_path, 'r'):
    #     if '\n' in line:
    #         one_addr = line[0:-1]
    #     else:
    #         one_addr = line
    #     address_14_list.append(one_addr)
    # addr_set = addr_set - set(address_14_list)
    # logger.info("2014 remove")
    # logger.info(len(addr_set))
    #
    # address_15_list = []
    # addr_path = "../wk/data_new/2015_addr.csv"
    # for line in open(addr_path, 'r'):
    #     if '\n' in line:
    #         one_addr = line[0:-1]
    #     else:
    #         one_addr = line
    #     address_15_list.append(one_addr)
    # addr_set = addr_set - set(address_15_list)
    # logger.info("2015 remove")
    # logger.info(len(addr_set))
    #
    # address_16_0_list = []
    # addr_path = "../wk/data_new/2016_6_30_addr.csv"
    # for line in open(addr_path, 'r'):
    #     if '\n' in line:
    #         one_addr = line[0:-1]
    #     else:
    #         one_addr = line
    #     address_16_0_list.append(one_addr)
    # addr_set = addr_set - set(address_16_0_list)
    # logger.info("2016 0 remove")
    # logger.info(len(addr_set))
    #
    # address_16_1_list = []
    # addr_path = "../wk/data_new/2016_12_31_addr.csv"
    # for line in open(addr_path, 'r'):
    #     if '\n' in line:
    #         one_addr = line[0:-1]
    #     else:
    #         one_addr = line
    #     address_16_1_list.append(one_addr)
    # addr_set = addr_set - set(address_16_1_list)
    # logger.info("2016 1 remove")
    # logger.info(len(addr_set))
    #
    # address_17_1_list = []
    # addr_path = "../wk/data_new/2017_addr_deduplication.csv"
    # for line in open(addr_path, 'r'):
    #     if '\n' in line:
    #         one_addr = line[0:-1]
    #     else:
    #         one_addr = line
    #     address_17_1_list.append(one_addr)
    # addr_set = addr_set - set(address_17_1_list)
    # logger.info("2017 remove")
    # logger.info(len(addr_set))
    #
    # address_18_1_list = []
    # addr_path = "../wk/data_new/2018_addr_deduplication.csv"
    # for line in open(addr_path, 'r'):
    #     if '\n' in line:
    #         one_addr = line[0:-1]
    #     else:
    #         one_addr = line
    #     address_18_1_list.append(one_addr)
    # addr_set = addr_set - set(address_18_1_list)
    # logger.info("2018 remove")
    # logger.info(len(addr_set))
    #
    # address_19_1_list = []
    # addr_path = "../wk/data_new/2019_addr_deduplication.csv"
    # for line in open(addr_path, 'r'):
    #     if '\n' in line:
    #         one_addr = line[0:-1]
    #     else:
    #         one_addr = line
    #     address_19_1_list.append(one_addr)
    # addr_set = addr_set - set(address_19_1_list)
    # logger.info("2019 remove")
    # logger.info(len(addr_set))
    #
    # address_20_1_list = []
    # addr_path = "../wk/data_new/2020_addr_deduplication.csv"
    # for line in open(addr_path, 'r'):
    #     if '\n' in line:
    #         one_addr = line[0:-1]
    #     else:
    #         one_addr = line
    #     address_20_1_list.append(one_addr)
    # addr_set = addr_set - set(address_20_1_list)
    # logger.info("2020 remove")
    # logger.info(len(addr_set))

    # addr_list = list(addr_set)



    process_list = []
    each_len = int(len(addr_list) / 10)
    for i in range(10):
        if i < 9:
            each_p_data = addr_list[i * each_len: i * each_len + each_len]
            process_X = Process(target=each_process, args=(each_p_data, i, db_tx, db_addr_tx,))
            process_list.append(process_X)
        else:
            each_p_data = addr_list[i * each_len:]
            process_X = Process(target=each_process, args=(each_p_data, i, db_tx, db_addr_tx,))
            process_list.append(process_X)

    for p in process_list:
        p.daemon = True
        p.start()

    for p in process_list:
        p.join()


def each_process(addr_data_list, p_id, db_tx, db_addr_tx):
    logger.info(str(p_id) + ' starting...')

    cnt = 0
    f = open("../nfs/tmk/www_experiment/result_csv/2021/features/" + str(p_id) + ".csv", "a+")
    csv_writer = csv.writer(f)
    csv_writer.writerow(
        ["address", "transaction_count", "receive_transaction_count", "receive_transaction_amount",
         "avg_receive_amount",
         "max_receive_amount", "min_receive_amount", "send_transaction_count", "send_transaction_amount",
         "avg_send_amount", "max_send_amount", "min_send_amount", "max_inputs_count", "min_inputs_count",
         "max_outputs_count", "min_outputs_count", "has_locktime", "active_period"])
    write_features_list = []
    for one_addr in addr_data_list:

        transaction_dict_list = []

        cnt += 1
        if cnt % 1000 == 0:
            logger.info(str(p_id) + '------' + str(cnt) + ' /// ' + str(len(addr_data_list)))

        addr_related_txs = db_addr_tx.get(bytes(one_addr, encoding='utf-8'))
        if addr_related_txs is not None:
            addr_related_txs = str(addr_related_txs, encoding='utf-8')
            addr_related_txs = fix_data_format(addr_related_txs)
            addr_related_txs = eval(addr_related_txs)
            for one_tx in addr_related_txs:
                one_tx = one_tx.split('/')
                this_tx_hash = one_tx[-1]
                tx_detail_info = db_tx.get(bytes(this_tx_hash, encoding='utf-8'))
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

        active_period = end_active_time - start_active_time
        write_features_list.append(
            [one_addr, transaction_count, receive_transaction_count, receive_transaction_amount,
             avg_receive_amount, max_receive_amount, min_receive_amount, send_transaction_count,
             send_transaction_amount, avg_send_amount, max_send_amount, min_send_amount, max_inputs_count,
             min_inputs_count, max_outputs_count, min_outputs_count, has_locktime, active_period])

        if len(write_features_list) > 1000:
            csv_writer.writerows(write_features_list)
            write_features_list = []
    csv_writer.writerows(write_features_list)


def combine_different_years(tag):
    txs_path = "../cd_data/related_txs/"
    combined_txs_path = "../nfs/RansomwareAddressDetection/combined_txs/"
    logger.info(tag + "start")
    tag_path = txs_path + tag + "/"
    year_list = os.listdir(tag_path)
    year_list = sorted(year_list)
    for year in year_list:
        logger.info(year)
        year_path = tag_path + year + "/"
        address_list = os.listdir(year_path)
        for address in address_list:
            command_str = "cat " + year_path + address + "  >>  " + combined_txs_path + tag + "/" + address
            os.system(command_str)
            # lines = []
            # with open(year_path + address, "r") as f:
            #     for line in f:
            #         lines.append(line)
            # with open(combined_txs_path + tag + "/" + address, "a+") as f:
            #     for line in lines:
            #         f.write(line)
    logger.info(tag + "finished")


if __name__ == '__main__':
    process_2016_addr_feature()

# if __name__ == "__main__":
#     out_txs_path = "../cd_data/related_txs/"
#     out_combined_txs_path = "../nfs/RansomwareAddressDetection/combined_txs/"
#     ransomware_tags_list = os.listdir(out_txs_path)
#
#     # 创建文件夹
#     for out_tag in ransomware_tags_list:
#         if os.path.exists(out_combined_txs_path + out_tag) is False:
#             os.mkdir(out_combined_txs_path + out_tag)
#     ps = []
#     for out_tag in ransomware_tags_list:
#         p = Process(target=combine_different_years, args=(out_tag,))
#         p.start()
#         ps.append(p)
#
#     for p in ps:
#         # 加入主进程
#         p.join()


# if __name__ == "__main__":
#     related_txs_path = "../cd_data/related_txs/"
#     related_addresses_path = "../wk/TWEB/ransomware_address_codes/AddressFeatures/related_address_tag.csv"
#     related_addresses_reader = csv.reader(open(related_addresses_path, 'r'))
#     address_tag_dict = dict()
#     related_address_set = set()
#     tags_set = set()
#     for line in related_addresses_reader:
#         address = line[0]
#         if address == "0000000000000000000000000000000000":
#             continue
#         tag = line[1]
#         tags_set.add(tag)
#         if address not in address_tag_dict:
#             address_tag_dict[address] = []
#         else:
#             if tag in address_tag_dict[address]:
#                 continue
#         address_tag_dict[address].append(tag)
#     related_address_set = set(address_tag_dict.keys())
#
#     for year in range(2009, 2022):
#         year = str(year)
#         for tag in tags_set:
#             tag_path = related_txs_path + tag + "/" + year
#             if os.path.exists(tag_path) is False:
#                 os.makedirs(tag_path)
#
#     txs_path = "../bitcoin_data_local_file/"
#     ps = list()
#     for year in range(2009, 2022):
#         year = str(year)
#         p = Process(target=find_txs_of_related_addresses, args=(year,))
#         p.start()
#         ps.append(p)
#
# for p in ps:
#     # 加入主进程
#     p.join()
