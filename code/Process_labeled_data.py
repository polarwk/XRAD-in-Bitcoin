import os
import sys
import csv
csv.field_size_limit(sys.maxsize)
from loguru import logger
import blocksci
import numpy as np

# chain = blocksci.Blockchain("../BlockSci/config_2.json")


def statistic_padua_addresses():
    path = "../nfs/wk/TDSC_major_revision/data/ransomware_dataset/padua/knowledge_base/addresses/"
    file_list = os.listdir(path)
    addresses = set()
    address_list = list()
    for file in file_list:
        file_path = path + file
        reader = csv.reader(open(file_path, "r"))
        for row in reader:
            addresses.add(row[0])
            address_list.append(row[0])
    writer = csv.writer(open("padua.csv", "a+"))
    for addr in addresses:
        writer.writerow([addr])
    for addr in addresses:
        if len(addr) < 20:
            logger.info(addr)
    logger.info(len(addresses))
    logger.info(len(address_list))
    return addresses


def statistic_montreal_addresses():
    path = "../nfs/wk/TDSC_major_revision/data/ransomware_dataset/montreal/dataset/blockchain/expanded_addresses_stats.csv"
    reader = csv.reader(open(path, "r"))
    addresses = list()
    address_set = set()
    name_set = set()
    for row in reader:
        addresses.append(row[0])
        address_set.add(row[0])
        name_set.add(row[1])

    logger.info(len(addresses))
    logger.info(len(address_set))
    logger.info(len(name_set))
    return address_set

def statistic_princeton_addresses():
    path = "../nfs/wk/TDSC_major_revision/data/ransomware_dataset/princeton/cerber-locky-addresses.csv"
    reader = csv.reader(open(path, "r"))
    addresses = set()
    address_list = list()
    for row in reader:
        addresses.add(row[4])
        address_list.append(row[4])
    
    for address in address_list:
        if len(address) < 10:
            logger.info(address)

    logger.info(len(addresses))
    logger.info(len(address_list))
    return addresses


def statistic_montreal_seed_addresses():

    path = "../nfs/wk/TDSC_major_revision/data/ransomware_dataset/montreal/ransomware-dataset/data/seed_addresses.csv"
    reader = csv.reader(open(path, "r"))
    names = list()
    name_set = set()
    addresses = set()
    for row in reader:
        names.append(row[1])
        name_set.add(row[1])
        addresses.add(row[0])

    logger.info(len(names))
    logger.info(len(name_set))
    return addresses

def statistic_bh_addresses():
    path = "../nfs/wk/TDSC_major_revision/data/ransomware_dataset/bitcoinheist/allAddresses.csv"
    reader = csv.reader(open(path, "r"))
    addresses = set()
    for row in reader:
        addresses.add(row[0])

    logger.info(len(addresses))
    return addresses


def check_seed_in_expanded():
    seed_path = "../nfs/wk/TDSC_major_revision/data/ransomware_dataset/montreal/ransomware-dataset/data/seed_addresses.csv"
    seed_addresses = set()
    reader = csv.reader(open(seed_path, "r"))
    for row in reader:
        seed_addresses.add(row[0])


    expanded_path = "../nfs/wk/TDSC_major_revision/data/ransomware_dataset/montreal/dataset/blockchain/expanded_addresses_stats.csv"
    reader = csv.reader(open(expanded_path, "r"))
    expanded_addresses = list()
    for row in reader:
        expanded_addresses.append(row[0])
    
    logger.info(len(expanded_addresses))

    # logger.info(len(seed_addresses-expanded_addresses))

    # for addr in seed_addresses-expanded_addresses:
    #     logger.info(addr)


def check_bh_involved_addresses():
    path = "../nfs/wk/TDSC_major_revision/data/ransomware_dataset/bitcoinheist/allAddresses.csv"
    reader = csv.reader(open(path, "r"))
    address_not_involved_txs = set()
    for row in reader:
        addr = row[0]
        addr_obj = chain.address_from_string(addr)
        if addr_obj is None:
            logger.info(addr)
            address_not_involved_txs.add(addr)

    logger.info(len(address_not_involved_txs))



def statistic_bh_with_white():
    path = "../nfs/wk/TDSC_major_revision/data/ransomware_dataset/bitcoinheist/BitcoinHeistData.csv"
    reader = csv.reader(open(path, "r"))
    addresses = set()
    for row in reader:
        if row[9] != "white":
            addresses.add(row[0])

    logger.info(len(addresses))



def combine_three_dataset():
    dataset1 = statistic_princeton_addresses()
    dataset2 = statistic_padua_addresses()
    dataset3 = statistic_montreal_addresses()
    logger.info(len(dataset1 | dataset2 | dataset3))


def combine_four_dataset():
    dataset1 = statistic_princeton_addresses()
    dataset2 = statistic_padua_addresses()
    dataset3 = statistic_montreal_addresses()
    dataset4 = statistic_montreal_seed_addresses()
    logger.info(len(dataset1 | dataset2 | dataset3 | dataset4))


def check_address_mul_labels_from_bh():
    path = "../nfs/wk/TDSC_major_revision/data/ransomware_dataset/bitcoinheist/allAddresses.csv"
    reader = csv.reader(open(path, "r"))
    address_mul_labels = dict()
    name_list = ["padua", "montreal", "princeton"]
    labels = set()
    for row in reader:
        length = len(row)
        addr = row[0]
        address_mul_labels[addr] = set()
        for i in range(1, length):
            for name in name_list:
                if name in row[i]:
                    label = row[i].replace(name, "")
                    address_mul_labels[addr].add(label)

    for addr, labels in address_mul_labels.items():
        if len(labels) > 1:
            logger.info(addr + "       " + str(labels))
        else:
            writer = csv.writer(open("ransomware_address.csv", "a+"))
            writer.writerow([addr, list(labels)[0]])


def ransom_distribution():
    dataset1 = statistic_princeton_addresses()
    dataset2 = statistic_padua_addresses()
    dataset3 = statistic_montreal_addresses()
    dataset4 = statistic_montreal_seed_addresses()
    all_addresses = dataset1 | dataset2 | dataset3

# def check_transactions_involved_addresses():
#     path = "ransomware_address.csv"
#     reader = csv.reader(open(path, "r"))
#     address_not_involved_txs = set()
#     for row in reader:
#         addr = row[0]
#         addr_obj = chain.address_from_string(addr)
#         if addr_obj is None:
#             logger.info(addr)
#             address_not_involved_txs.add(addr)
#         else:
#             writer = csv.writer(open("final_ransomware_address.csv", "a+"))
#             writer.writerow(row)

#     logger.info(len(address_not_involved_txs))


def load_padua_addresses():
    path = "../ransomware_dataset/padua/knowledge_base/addresses/"
    file_list = os.listdir(path)
    address_label = dict()
    for file in file_list:
        name = file.split("_addresses")[0]
        file_path = path + file
        reader = csv.reader(open(file_path, "r"))
        for row in reader:
            address_label[row[0]] = name
    return address_label


def load_montreal_addresses():
    path = "../ransomware_dataset/montreal/dataset/blockchain/expanded_addresses_stats.csv"
    reader = csv.reader(open(path, "r"))
    address_label = dict()
    for row in reader:
        address_label[row[0]] = row[1]

    return address_label

def load_princeton_addresses():
    path = "../ransomware_dataset/princeton/cerber-locky-addresses.csv"
    reader = csv.reader(open(path, "r"))
    address_label = dict()
    next(reader)
    for row in reader:
        address_label[row[4]] = row[1]

    return address_label

def combine_address_with_label():
    padua_address_label = load_padua_addresses()
    montral_address_label = load_montreal_addresses()
    princeton_address_label = load_princeton_addresses()
    rs_addr_label = dict()
    rs_addr_label.update(padua_address_label)
    rs_addr_label.update(montral_address_label)
    rs_addr_label.update(princeton_address_label)
    logger.info(len(rs_addr_label))

    writer = csv.writer(open("../ransomware_dataset/all_rs_addresses_single_label.csv", "a+"))
    for addr, label in rs_addr_label.items():
        writer.writerow([addr, label])

def check_transactions_involved_addresses():
    path = "../ransomware_dataset/all_rs_addresses_single_label.csv"
    reader = csv.reader(open(path, "r"))
    address_not_involved_txs = set()
    for row in reader:
        addr = row[0]
        addr_obj = chain.address_from_string(addr)
        if addr_obj is None or len(addr_obj.txes.to_list()) == 0:
            logger.info(addr)
            address_not_involved_txs.add(addr)
        else:
            writer = csv.writer(open("../ransomware_dataset/final_rs_addresses_single_label.csv", "a+"))
            writer.writerow(row)

    logger.info(len(address_not_involved_txs))

def load_all_rs_addr_label():
    addr_label = dict()
    reader = csv.reader(open("../ransomware_dataset/rs_addr_clustered.csv", "r"))
    for row in reader:
        addr_label[row[0]] = row[1]
    
    return addr_label


def add_rs_label_to_cascade_features():
    all_rs_addr_label = load_all_rs_addr_label()
    reader = csv.reader(open("../addr_cascade_feature/ransomware_addr_cascade_feature.csv", "r"))
    writer = csv.writer(open("../addr_cascade_feature/ransomware_addr_cascade_feature_with_label.csv", "a+"))

    for row in reader:
        label = all_rs_addr_label[row[0]]
        row.insert(1, label)
        writer.writerow(row)


# 计算每个勒索活动的活跃时间段
def activate_time_period():
    rs_path = "../ransomware_dataset/final_rs_addresses_single_label.csv"
    reader = csv.reader(open(rs_path, "r"))
    time_period = dict()
    for row in reader:
        addr = row[0]
        label = row[1]
        if label not in time_period:
            time_period[label] = [sys.maxsize, 0]
        addr_object = chain.address_from_string(addr)
        for tx in addr_object.txes.to_list():
            if time_period[label][0] > tx.block_height:
                time_period[label][0] = tx.block_height
            if time_period[label][1] < tx.block_height:
                time_period[label][1] = tx.block_height
    writer = csv.writer(open("../ransomware_dataset/ransomware_addr_active_time_period.csv", "a+"))
    for label, period in time_period.items():
        writer.writerow([label, period[0], period[1], chain.blocks[period[0]].time, chain.blocks[period[1]].time])


# 统计每个地址的活跃时间
def statistic_address_txs_num():
    path = "../ransomware_dataset/rs_addr_clustered.csv"
    reader = csv.reader(open(path, "r"))

    writer = csv.writer(open("../ransomware_dataset/rs_addr_clustered_details.csv", "a+"))

    for row in reader:
        addr = row[0]
        label = row[1]
        addr_object = chain.address_from_string(addr)
        txs_count = len(addr_object.txes.to_list())
        start_height = sys.maxsize
        end_height = 0
        for tx in addr_object.txes:
            if start_height > tx.block_height:
                start_height = tx.block_height
            if end_height < tx.block_height:
                end_height = tx.block_height
        writer.writerow([addr, label, txs_count, start_height, end_height, chain.blocks[start_height].time, chain.blocks[end_height].time])

# 统计每个勒索家族的地址个数
def statistic_family_addr_num():
    path = "../nfs/wk/TDSC_major_revision/data/wk/TDSC_major_revision/ransomware_dataset/rs_addr_clustered.csv"
    reader = csv.reader(open(path, "r"))
    family_count = dict()
    for row in reader:
        if row[1] == "DMALockerv3" or row[1] == "DMA-Locker":
            row[1] = "DMALocker"
        if row[1] == "XLockerv5.0":
            row[1] = "XLocker"
        if row[1] == "Globev3":
            row[1] = "Globe"
        if row[1] not in family_count:
            family_count[row[1]] = 0

        family_count[row[1]] += 1
    
    logger.info(len(family_count))

    for family, count in family_count.items():
        logger.info(family + ":" + str(count))
    
    logger.info("sorted--------------------------------")

    family_count_tuple = list(sorted(family_count.items(), key=lambda x: x[1], reverse=True))
    family_count = len(family_count_tuple)
    for i in range(0, family_count, 3):
        if i != family_count-2:
            print(family_count_tuple[i][0] + "   &   " + str(family_count_tuple[i][1]) + "    &   " +  family_count_tuple[i+1][0] + "    &    " +str(family_count_tuple[i+1][1])  + "    &   " +  family_count_tuple[i+2][0] + "    &    " +str(family_count_tuple[i+2][1]) + "  \\\\")
        else:
            print(family_count_tuple[i][0] + "   &   " + str(family_count_tuple[i][1]) + "    &   " +  family_count_tuple[i+1][0] + "    &    " +str(family_count_tuple[i+1][1])  +  "  \\\\")
        
    # for family, count in family_count_tuple:
    #     print(family + "   &   " + str(count) + "    \\\\")

# 查看每个勒索家族地址收入的分布情况
def statistic_ransom_distribution():
    # path = "../ransomware_dataset/final_rs_addresses_single_label.csv"
    # reader = csv.reader(open(path, "r"))
    # family_revenue = dict()
    # for row in reader:
    #     addr = row[0]
    #     label = row[1]
    #     if label not in family_revenue:
    #         family_revenue[label] = []
    #     addr_object = chain.address_from_string(addr)
    #     for output in addr_object.outputs:
    #         family_revenue[label].append(output.value)
    
    # for label, value_list in family_revenue.items():
    #     logger.info(label + "  :  " + str(np.var(value_list)))
    # path = "../ransomware_dataset/final_rs_addresses_single_label.csv"
    # reader = csv.reader(open(path, "r"))
    # family_revenue = dict()
    # addr_label = dict()
    # for row in reader:
    #     addr = row[0]
    #     label = row[1]
    #     addr_label[addr] = label
    # for addr, label in addr_label.items():
    #     if label not in family_revenue:
    #         family_revenue[label] = []
    #     addr_object = chain.address_from_string(addr)
    #     # family_revenue[label][addr] = []
    #     jump = False
    #     for tx in addr_object.output_txes:
    #         for tx_input in tx.inputs:
    #             if tx_input.address.address_string in addr_label:
    #                 jump = True
    #                 break
    #         if jump:
    #             break
    #     if jump:
    #         continue
        
    #     for output in addr_object.outputs:
    #         family_revenue[label].append(output.value)
    
    # for label, value_list in family_revenue.items():
    #     logger.info(label + "  :  " + str(np.var(value_list)))

    all_addr_label = load_all_rs_addr_label()

    path = "../ransomware_dataset/rs_addr_extra_from_clustering.csv"
    reader = csv.reader(open(path, "r"))
    family_revenue = dict()
    addr_label = dict()
    for row in reader:
        addr = row[0]
        label = row[1]
        addr_label[addr] = label
    for addr, label in addr_label.items():
        if label not in family_revenue:
            family_revenue[label] = dict()
        addr_object = chain.address_from_string(addr)
        family_revenue[label][addr] = []
        jump = False
        # logger.info(len(addr_object.output_txes.to_list()))
        for tx in addr_object.output_txes:

            for tx_input in tx.inputs:
                logger.info(tx_input.address.address_string)
                if tx_input.address.address_string in all_addr_label:
                    jump = True
                    break
            if jump:
                break
        if jump:
            continue

        for output in addr_object.outputs:
            family_revenue[label][addr].append(output.value)
    
    # for label, value_pair in family_revenue.items():
    #     logger.info(label)
    #     for addr, value_list in value_pair.items():
    #         logger.info(addr + "    :   " + str(value_list))
    
    writer = csv.writer(open("extra_addr_value.csv", "a+"))
    for label, value_pair in family_revenue.items():
        writer.writerow([label])
        for addr, value_list in value_pair.items():
            value_list.insert(0, addr)
            writer.writerow(value_list)
    
# 加载walletexplorer网站的数据
def load_walletexplorer_address():
    path = "../walletexplorer.csv"
    addr_label = dict()
    reader = csv.reader(open(path, "r"))
    for row in reader:
        addr_label[row[0]] = row[1]
    
    return addr_label
    
# 收集的勒索地址中有些是交易所地址，需要进行过滤。
def statistic_ransomware_address_in_walletexplorer():
    walletexplorer_addr_label = load_walletexplorer_address()
    path = "../ransomware_dataset/final_rs_addresses_single_label.csv"
    reader = csv.reader(open(path, "r"))
    count = 0
    writer = csv.writer(open("../ransomware_dataset/wrong_tag_ransomware.csv", "a+"))

    for row in reader:
        if row[0] in walletexplorer_addr_label:
            if row[1] == walletexplorer_addr_label[row[0]]:
                continue
            logger.info(row[0] + ":" + row[1] + ":" + walletexplorer_addr_label[row[0]])
            writer.writerow([row[0], row[1],  walletexplorer_addr_label[row[0]]])
            count += 1


    logger.info(count)

# 过滤勒索地址中为交易所的地址
def filter_ransomware_address_in_walletexplorer():
    walletexplorer_addr_label = load_walletexplorer_address()
    path = "../ransomware_dataset/final_rs_addresses_single_label.csv"
    reader = csv.reader(open(path, "r"))
    writer = csv.writer(open("../ransomware_dataset/final_ransomware_addr.csv", "a+"))

    for row in reader:
        if row[0] in walletexplorer_addr_label and row[1] != walletexplorer_addr_label[row[0]]:
            continue
        writer.writerow(row)


# 过滤勒索地址特征
def filter_ransomware_cascade_feature():
    walletexplorer_addr_label = load_walletexplorer_address()

    ransomware_cascade_path = "../addr_cascade_feature/ransomware_addr_cascade_feature_with_label.csv"
    reader = csv.reader(open(ransomware_cascade_path, "r"))

    write_path = "../addr_cascade_feature/ransomware_addr_cascade_feature_with_label_filter.csv"
    writer = csv.writer(open(write_path, "a+"))

    for row in reader:
        if row[0] in walletexplorer_addr_label and row[1] != walletexplorer_addr_label[row[0]]:
            continue
        writer.writerow(row)




if __name__ == '__main__':
    statistic_family_addr_num()
    # check_transactions_involved_addresses()
    # add_rs_label_to_cascade_features()
    # activate_time_period()
    # statistic_address_txs_num()
    # statistic_family_addr_num()
    # statistic_ransom_distribution()
    # statistic_family_addr_num()
    # statistic_address_txs_num()
    # statistic_padua_addresses()
    # statistic_princeton_addresses()
    # check_seed_in_expanded()
    # statistic_montreal_seed_addresses()
    # statistic_address_txs_num()
    # statistic_ransom_distribution()
    # filter_ransomware_cascade_feature()