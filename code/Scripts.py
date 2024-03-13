import os
import sys
import csv
from loguru import logger
import blocksci
# import rocksdb
import requests
import time
import numpy as np
import rocksdb
from multiprocess import Process

csv.field_size_limit(sys.maxsize)
from tqdm import tqdm

blockchain = blocksci.Blockchain("../BlockSci/config_2.json")


# addr_oppos_db = rocksdb.DB("../addr_oppos/addr_oppos.db", rocksdb.Options(create_if_missing=True), read_only=True)


# 查看勒索地址会不会同时出现在一笔交易的输入和输出
def check_in_and_out(addr: str):
    addr_object = blockchain.address_from_string(addr)
    As_input_tx_hashs = set()
    As_output_tx_hashs = set()
    for tx in addr_object.input_txes:
        As_input_tx_hashs.add(str(tx.hash))

    for tx in addr_object.output_txes:
        As_output_tx_hashs.add(str(tx.hash))

    overlap = As_input_tx_hashs & As_output_tx_hashs
    if len(overlap) > 0:
        logger.info(addr)
        logger.info(overlap)
        return True
    return False


def test_blocksci():
    addr_object = blockchain.address_from_string("1MdYC22Gmjp2ejVPCxyYjFyWbQCYTGhGq8")
    logger.info(addr_object.type is blocksci.address_type.pubkeyhash)
    exit(0)
    # logger.info(addr_object.raw_type)
    # logger.info(addr_object.type)
    # logger.info(type(addr_object))
    # logger.info(type(addr_object) == blocksci.PubkeyHashAddress)
    # exit(0)
    tx = blockchain.tx_with_hash("dbaf14e1c476e76ea05a8b71921a46d6b06f0a950f17c5f9f1a03b8fae467f10")
    logger.info(tx.outputs.to_list()[0].address)

    logger.info(tx.outputs.to_list()[0].address == addr_object)
    # logger.info(addr_object)
    # logger.info(addr_object.balance())
    # logger.info(addr_object.address_num)
    # logger.info(len(addr_object.txes.to_list()))
    # for block in tqdm(blockchain.blocks):
    #     for tx in block.txes.to_list():
    #         for address in tx.inputs.address.to_list():
    #             if type(address) is blocksci.WitnessUnknownAddress:
    #                 logger.info(tx.hash)
    #                 logger.info(address)

    #         for address in tx.outputs.address.to_list():
    #             if type(address) is blocksci.WitnessUnknownAddress:
    #                 logger.info(tx.hash)
    #                 logger.info(address)


def check_random_in_ransomware():
    ransomware_path = "../ransomware_dataset/final_ransomware_address.csv"
    random_path = "../random_addr_1000000_1.csv"

    ransomware_address = set()
    reader = csv.reader(open(ransomware_path, "r"))
    for row in reader:
        ransomware_address.add(row[0])

    random_address = set()
    reader = csv.reader(open(random_path, "r"))
    for row in reader:
        random_address.add(row[0])

    logger.info(len(random_address.intersection(ransomware_address)))


# 去掉本身是勒索地址的随机地址。
def remove_random_addr_1_in_ransomware():
    random_1_cascade_path = "../addr_cascade_feature/random_addr_1_cascade_feature.csv"

    ransomware_path = "../ransomware_dataset/final_ransomware_address.csv"
    ransomware_address = set()
    reader = csv.reader(open(ransomware_path, "r"))
    for row in reader:
        ransomware_address.add(row[0])

    reader = csv.reader(open(random_1_cascade_path, "r"))
    writer = csv.writer(
        open("../addr_cascade_feature/random_addr_1_cascade_feature_filtered.csv",
             "a+"))
    for row in reader:
        if row[0] in ransomware_address:
            continue
        writer.writerow(row)


# 将之前爬取的walletexplorer标签数据整合成一个文件
def combine_walletexplorer():
    folder = "../nfs/backup/entity/"
    industry_list = os.listdir(folder)
    writer = csv.writer(open("../walletexplorer.csv", "a+"))
    for industry in industry_list:
        industry_path = folder + industry + "/"
        entity_list = os.listdir(industry_path)
        for entity in entity_list:
            logger.info(entity)
            entity_path = industry_path + entity
            entity_name = entity.split(".csv")[0]
            reader = csv.reader(open(entity_path, "r"))
            for row in reader:
                writer.writerow([row[0], entity_name])
    logger.info("finish")


# 在地址自身特征上添加地址类型，以及邻居兄弟的个数
def add_extra_features_to_no_label():
    addr_oppos_path = "../addr_oppos/addr_oppos.db"
    addr_oppos_db = rocksdb.DB(addr_oppos_path, rocksdb.Options(create_if_missing=True), read_only=True)

    origin_cascade_features_path_list = [
        "../addr_cascade_feature/random_addr_1_cascade_feature_filtered.csv",
        "../addr_cascade_feature/random_addr_2_cascade_feature.csv"]
    for feature_path in origin_cascade_features_path_list:
        logger.info(feature_path)
        name = feature_path.split("/")[-1].split(".csv")[0]
        name = name + "_extra.csv"
        writer = csv.writer(open("../addr_cascade_feature/" + name, "a+"))
        reader = csv.reader(open(feature_path, "r"))
        features_list = []
        for row in reader:
            features_list.append(row)
        for row in tqdm(features_list):
            addr = row[0]
            features = row[1:]
            addr_object = blockchain.address_from_string(addr)
            # 添加地址类型
            features.insert(0, addr_object.raw_type)

            addr_oppos = addr_oppos_db.get(bytes(addr, encoding='utf-8'))
            addr_oppos = eval(str(addr_oppos, encoding='utf-8'))
            for index, role in enumerate(addr_oppos):
                features.insert(index + 3, len(role))
            features.insert(0, addr)
            writer.writerow(features)


# 在地址自身特征上添加地址类型，以及邻居兄弟的个数
def add_extra_features_to_with_label():
    addr_oppos_path = "../addr_oppos/addr_oppos.db"
    addr_oppos_db = rocksdb.DB(addr_oppos_path, rocksdb.Options(create_if_missing=True), read_only=True)

    origin_cascade_features_path_list = [
        "../addr_cascade_feature/ransomware_addr_cascade_feature_with_label.csv"]
    for feature_path in origin_cascade_features_path_list:
        logger.info(feature_path)
        name = feature_path.split("/")[-1].split(".csv")[0]
        name = name + "_extra.csv"
        writer = csv.writer(open("../addr_cascade_feature/" + name, "a+"))
        reader = csv.reader(open(feature_path, "r"))
        features_list = []
        for row in reader:
            features_list.append(row)
        for row in tqdm(features_list):
            addr = row[0]
            features = row[1:]
            addr_object = blockchain.address_from_string(addr)
            # 添加地址类型
            features.insert(1, addr_object.raw_type)

            addr_oppos = addr_oppos_db.get(bytes(addr, encoding='utf-8'))
            addr_oppos = eval(str(addr_oppos, encoding='utf-8'))
            for index, role in enumerate(addr_oppos):
                features.insert(index + 4, len(role))
            features.insert(0, addr)
            writer.writerow(features)


# 统计每个勒索家族的地址个数


# 过滤掉addr_value中的交易所地址
def filter_exchange():
    true_addr_set = set()
    writer = csv.writer(open("../nfs/wk/TDSC_major_revision/code/ransom_addr.csv", "a+"))
    reader = csv.reader(open("../ransomware_dataset/final_ransomware_addr.csv", "r"))
    for row in reader:
        true_addr_set.add(row[0])
    reader = csv.reader(open("../nfs/wk/TDSC_major_revision/code/addr_value.csv", "r"))
    for row in reader:
        if (len(row) == 1):
            writer.writerow(row)
        elif (len(row) > 1 and row[0] in true_addr_set):
            writer.writerow(row)


# 根据金额过滤掉交易所地址
# def filter_with_heurstics():
#     result_list=[]
#     reader = csv.reader(open("../nfs/wk/TDSC_major_revision/code/ransom_addr.csv", "r"))
#     for row in reader:
#         if(len(row)<5):
#             for i in range(1,len(row)):
#               value=row[i]
#               if(value[-1]!='0'):
#                 result_list.append(row)
#                 break
#     writer = csv.writer(open("../nfs/wk/TDSC_major_revision/code/suspect_ransom_addr.csv", "w+"))
#     for item in result_list:
#         writer.writerow(item)

# 获取地址所属勒索家族映射集合
def get_address_family():
    addr_family = {}
    tag = '7456'
    reader = csv.reader(open("../nfs/wk/TDSC_major_revision/code/ransom_addr.csv", "r"))
    for row in reader:
        if (len(row) == 1 and len(row[0]) < 20):
            tag = row[0]
        else:
            addr_family[row[0]] = tag

    # logger.info(addr_family['1PeFqJwKUWmdM225Gv8MnjhbzbXShzy9Kg'])
    return addr_family


# 根据金额启发式过滤疑似非勒索地址的地址
def filter_with_heurstics():
    addr_dict = get_address_family()
    result_list = []
    reader = csv.reader(open("../nfs/wk/TDSC_major_revision/code/ransom_addr.csv", "r"))
    for row in reader:
        if (len(row) < 5):
            for i in range(1, len(row)):
                value = row[i]
                if (value[-1] != '0'):
                    row.append(addr_dict[row[0]])
                    result_list.append(row)
                    break
    writer = csv.writer(open("../nfs/wk/TDSC_major_revision/code/suspect_ransom_addr.csv", "w+"))
    for item in result_list:
        writer.writerow(item)


def filter_with_api_160():
    filepath = '../ransomware_dataset/final_rs_addresses_single_label.csv'
    reader = csv.reader(open(filepath, "r"))

    write_path = "../nfs/wk/TDSC_major_revision/code/api_address-160.csv"

    addresses = list()
    for row in reader:
        addresses.append(row)

    addresses = addresses[0:5000]

    for row in tqdm(addresses):
        url = "http://www.walletexplorer.com/api/1/address?address=" + row[0] + "&from=0&count=100&caller=polarwk"
        tx_req = requests.get(url=url)
        result = eval(tx_req.text.replace("true", "True").replace("false", "False"))
        if ('label' in result):
            f = open(write_path, "a+")
            writer = csv.writer(f)
            row.append(result['label'])
            writer.writerow(row)
            f.close()
        time.sleep(6)


def filter_with_api_70():
    filepath = '../ransomware_dataset/final_rs_addresses_single_label.csv'
    reader = csv.reader(open(filepath, "r"))

    write_path = "../nfs/wk/TDSC_major_revision/code/api_address-70.csv"

    addresses = list()
    for row in reader:
        addresses.append(row)

    addresses = addresses[-5098:-5000]

    for row in tqdm(addresses):
        url = "http://www.walletexplorer.com/api/1/address?address=" + row[
            0] + "&from=0&count=100&caller=20110240077@fudan.edu.cn"
        tx_req = requests.get(url=url)
        result = eval(tx_req.text.replace("true", "True").replace("false", "False"))
        if ('label' in result):
            f = open(write_path, "a+")
            writer = csv.writer(f)
            row.append(result['label'])
            writer.writerow(row)
            f.close()
        time.sleep(5)


def filter_with_api_55():
    filepath = '../nfs/wk/TDSC_major_revision/data/ransomware_dataset/final_rs_addresses_single_label.csv'
    reader = csv.reader(open(filepath, "r"))

    write_path = "../nfs/wk/TDSC_major_revision/code/api_address-55.csv"

    addresses = list()
    for row in reader:
        addresses.append(row)

    addresses = addresses[-10724:-10000]

    for row in tqdm(addresses):
        url = "http://www.walletexplorer.com/api/1/address?address=" + row[
            0] + "&from=0&count=100&caller=48273647235@qq.com"
        tx_req = requests.get(url=url)
        result = eval(tx_req.text.replace("true", "True").replace("false", "False"))
        if ('label' in result):
            f = open(write_path, "a+")
            writer = csv.writer(f)
            row.append(result['label'])
            writer.writerow(row)
            f.close()
        time.sleep(5)


def filter_with_api_56():
    filepath = '../nfs/wk/TDSC_major_revision/data/ransomware_dataset/final_rs_addresses_single_label.csv'
    reader = csv.reader(open(filepath, "r"))

    write_path = "../nfs/wk/TDSC_major_revision/code/api_address-56.csv"

    addresses = list()
    for row in reader:
        addresses.append(row)

    addresses = addresses[-18691:-15000]

    for row in tqdm(addresses):
        url = "http://www.walletexplorer.com/api/1/address?address=" + row[
            0] + "&from=0&count=100&caller=34534634243@fudan.edu.cn"
        tx_req = requests.get(url=url)
        result = eval(tx_req.text.replace("true", "True").replace("false", "False"))
        if ('label' in result):
            f = open(write_path, "a+")
            writer = csv.writer(f)
            row.append(result['label'])
            writer.writerow(row)
            f.close()
        time.sleep(4)


def get_proxy():
    return requests.get("http://127.0.0.1:5010/get/").json()


def delete_proxy(proxy):
    requests.get("http://127.0.0.1:5010/delete/?proxy={}".format(proxy))


def filter_with_api_160_pool():
    filepath = '../nfs/wk/TDSC_major_revision/data/ransomware_dataset/final_rs_addresses_single_label.csv'
    reader = csv.reader(open(filepath, "r"))

    write_path = "../nfs/wk/TDSC_major_revision/code/api_address-160-pool.csv"

    addresses = list()
    for row in reader:
        addresses.append(row)

    # addresses = addresses[-19382:-15000]

    for row in tqdm(addresses):
        proxy = get_proxy().get("proxy")
        url = "http://www.walletexplorer.com/api/1/address?address=" + row[
            0] + "&from=0&count=100&caller=43893253634@qq.com"
        tx_req = requests.get(url=url, proxies={"http": "http://{}".format(proxy)})
        result = eval(tx_req.text.replace("true", "True").replace("false", "False"))
        if ('label' in result):
            f = open(write_path, "a+")
            writer = csv.writer(f)
            row.append(result['label'])
            writer.writerow(row)
            f.close()
        # time.sleep(5)


#    
def filter_with_api():
    f = open("../nfs/wk/TDSC_major_revision/code/api_address.csv", "a+")
    writer = csv.writer(f)

    logger.info("start")
    count = 0

    filepath = '../ransomware_dataset/final_rs_addresses_single_label.csv'
    reader = csv.reader(open(filepath, "r"))

    for row in reader:
        address = row[0]
        if (count < 1450):
            count = count + 1
            continue

        url = "http://www.walletexplorer.com/api/1/address?address=" + address + "&from=0&count=100&caller=20110240034@fudan.edu.cn"
        tx_req = requests.get(url=url)
        result = eval(tx_req.text.replace("true", "True").replace("false", "False"))
        if ('label' in result):
            row.append(result['label'])
            writer.writerow(row)
        count = count + 1
        time.sleep(1.5)
        if (count % 50 == 0):
            logger.info(count)
            f.close()
            f = open("../nfs/wk/TDSC_major_revision/code/api_address.csv", "a+")
            writer = csv.writer(f)


# 合并被过滤的非勒索地址
def combine_filtered_ransomware_address():
    folder = "../nfs/wk/TDSC_major_revision/code/"
    file_list = ["api_address-160.csv", "api_address-160-pool.csv", "api_address-55.csv", "api_address-56.csv",
                 "api_address.csv", "api_address_old_2.csv", "api_address_old.csv"]
    addr_label = dict()
    for file_name in file_list:
        file_path = folder + file_name
        reader = csv.reader(open(file_path, "r"))
        for row in reader:
            addr = row[0]
            label = row[1:]
            if label[0] == label[1]:
                continue
            addr_label[addr] = label
    writer = csv.writer(open("../ransomware_dataset/wrong_labeled_rs_addr.csv", "a+"))
    for addr, label in addr_label.items():
        writer.writerow([addr, label[0], label[1]])

    logger.info(len(addr_label))


# 将总的勒索地址集合删除掉walletexplorer
def remove_walletexplorer_tag_addr_from_final():
    path = "../ransomware_dataset/wrong_labeled_rs_addr.csv"
    reader = csv.reader(open(path, "r"))
    walletexplorer_addr = set()
    for row in reader:
        walletexplorer_addr.add(row[0])

    path = "../ransomware_dataset/final_rs_addresses_single_label.csv"
    reader = csv.reader(open(path, "r"))
    write_path = "../ransomware_dataset/final_rs_addresses_filtered.csv"
    writer = csv.writer(open(write_path, "a+"))
    for row in reader:
        if row[0] in walletexplorer_addr:
            continue
        writer.writerow(row)


# 将提取的勒索地址级联特征，进行过滤
def filter_ransomware_cascade_feature():
    path = "../ransomware_dataset/wrong_labeled_rs_addr.csv"
    reader = csv.reader(open(path, "r"))
    walletexplorer_addr = set()
    for row in reader:
        walletexplorer_addr.add(row[0])

    path = "../addr_cascade_feature/ransomware_addr_cascade_feature_with_label.csv"
    reader = csv.reader(open(path, "r"))

    write_path = "../addr_cascade_feature/final_ransomware_addr_cascade_feature_with_label.csv"
    writer = csv.writer(open(write_path, "a+"))

    for row in reader:
        if row[0] in walletexplorer_addr:
            continue
        writer.writerow(row)


# 从self_feature中删除一些特征，相应地，级联特征中对应位置的也要删除
def remove_features():
    remove_self_index_list = [2, 3, 4, 5, 7, 8, 9, 10]
    remove_index_list = [2, 3, 4, 5, 7, 8, 9, 10]
    for i in range(4):
        for index in remove_self_index_list:
            for j in range(4):
                remove_index_list.append(18 + 18 * 4 * i + 4 * index + j)

    # unlabeled_feature_path = "../addr_cascade_feature/random_addr_1_cascade_feature_filtered.csv"
    # reader = csv.reader(open(unlabeled_feature_path, "r"))
    # save_unlabeled_feature_path = "../addr_cascade_feature/random_addr_1_cascade_feature_filtered_remove_amount.csv"
    # writer = csv.writer(open(save_unlabeled_feature_path, "a+"))
    # for row in reader:
    #     addr = row[0]
    #     features = row[1:]
    #     new_features = np.delete(features, remove_index_list).tolist()
    #     new_features.insert(0, addr)
    #     writer.writerow(new_features)

    ransomware_feature_path = "../addr_cascade_feature/rs_addr_extra_from_clustering.csv"
    reader = csv.reader(open(ransomware_feature_path, "r"))
    save_ransomware_feature_path = "../addr_cascade_feature/rs_addr_extra_from_clustering_remove_amount.csv"
    writer = csv.writer(open(save_ransomware_feature_path, "a+"))

    for row in reader:
        addr = row[0]
        label = row[1]
        features = row[2:]
        new_features = np.delete(features, remove_index_list).tolist()
        new_features.insert(0, addr)
        new_features.insert(1, label)
        writer.writerow(new_features)


# 加载扩充前，并利用walletexplorer网站检测后的勒索地址
def load_ransomware_addr_before_cluster():
    path = "../ransomware_dataset/final_rs_addresses_filtered.csv"
    reader = csv.reader(open(path, "r"))
    rs_addr = set()
    addr_label = dict()
    for row in reader:
        rs_addr.add(row[0])
        addr_label[row[0]] = row[1]

    return rs_addr, addr_label


# 地址聚类后，查看每个勒索家族的地址数量, 并将每个扩充来的地址利用walletexplorer网站进行验证
def statistic_family_addr_clustered():
    rs_addr, addr_label = load_ransomware_addr_before_cluster()
    path = "../nfs/tmk/yxt/1000/quchong_new/"
    rs_name_list = os.listdir(path)
    clustered_addr = set()
    clustered_addr_label = dict()
    for rs_name in rs_name_list:
        rs_path = path + rs_name
        reader = csv.reader(open(rs_path, "r"))
        for row in reader:
            clustered_addr.add(row[0])
            clustered_addr_label[row[0]] = rs_name.split(".csv")[0]
    writer = csv.writer(open("../ransomware_dataset/rs_addr_clustered.csv", "a+"))
    for addr, label in clustered_addr_label.items():
        writer.writerow([addr, label])

    logger.info(len(clustered_addr - rs_addr))

    # logger.info(clustered_addr - rs_addr)
    extra_writer = csv.writer(
        open("../ransomware_dataset/rs_addr_extra_from_clustering.csv", "a+"))
    extra_addr = clustered_addr - rs_addr
    for addr in extra_addr:
        extra_writer.writerow([addr, clustered_addr_label[addr]])
    # write_path = "api_address_extra.csv"
    # for row in tqdm(extra_addr):
    #     url="http://www.walletexplorer.com/api/1/address?address="+row[0]+"&from=0&count=100&caller=48273647235@qq.com"
    #     tx_req = requests.get(url=url)
    #     result=eval(tx_req.text.replace("true","True").replace("false","False"))
    #     if('label' in result):
    #         f = open(write_path, "a+")
    #         writer = csv.writer(f)
    #         row.append(result['label'])
    #         writer.writerow(row)
    #         f.close()
    #     time.sleep(6)


# 过滤出交易次数大于50的勒索地址
def filter_ransomware_address_more_10():
    path = "../ransomware_dataset/rs_addr_clustered.csv"
    reader = csv.reader(open(path, "r"))
    count = 0
    writer = csv.writer(open("txs-50.csv", "a+"))
    for row in reader:
        addr_object = blockchain.address_from_string(row[0])
        if len(addr_object.txes.to_list()) > 50:
            count += 1
            logger.info(row)
            writer.writerow(row)

    logger.info(count)


# 添加一些勒索地址很明显的特征
def add_features_to_addr(addr: str):
    # 地址类型、交易对手个数[分四类还是直接一类]、交易金额整数位数、这个地址是否同时出现在交易输入和交易输出
    addr_object = blockchain.address_from_string(addr)
    features = []
    features.append(addr_object.raw_type)
    oppos = addr_oppos_db.get(bytes(addr, encoding='utf-8'))
    oppos = str(oppos, encoding='utf-8')
    oppos = eval(oppos)
    for role_oppo in oppos:
        features.append(len(role_oppo))
    max_zero_count = 0
    min_zero_count = sys.maxsize
    for value in list(addr_object.outputs.value):
        zero_count = 0
        for i in str(value)[::-1]:
            if i == '0':
                zero_count += 1
            else:
                break
        if zero_count > max_zero_count:
            max_zero_count = zero_count
        if zero_count < min_zero_count:
            min_zero_count = zero_count

    features.append(max_zero_count)
    features.append(min_zero_count)

    self_change = 0
    for tx in addr_object.txes.to_list():
        input_addrs = tx.inputs.address.to_list()
        output_addrs = tx.outputs.address.to_list()
        if addr_object in input_addrs and addr_object in output_addrs:
            self_change = 1
            break
    if self_change:
        features.append(1)
    else:
        features.append(0)

    return features


# 添加勒索地址的特征
def add_feature_to_ransomware_addr():
    path = "../ransomware_dataset/rs_addr_clustered.csv"
    write_path = "../addr_self_feature/rs_addr_added_features.csv"
    writer = csv.writer(open(write_path, "a+"))
    rs_addr = list()
    reader = csv.reader(open(path, "r"))
    for row in reader:
        rs_addr.append(row[0])

    for addr in tqdm(rs_addr):
        features = add_features_to_addr(addr)
        features.insert(0, addr)
        writer.writerow(features)


# 添加特征到一个list中的地址
def add_feature_to_list_addr(addrs: list, pid: int):
    writer = csv.writer(
        open("../addr_self_feature/random_addr_1_added_feature/" + str(pid) + ".csv",
             "a+"))
    for addr in tqdm(addrs):
        features = add_features_to_addr(addr)
        features.insert(0, addr)
        writer.writerow(features)

    logger.info(str(pid) + "   :   finish")


# 添加特征到未标记地址
def add_feature_to_random_1_addr():
    path_list = ["../random_addr_1000000_1.csv"]
    unlabeled_addr = list()
    for path in path_list:
        reader = csv.reader(open(path, "r"))
        for row in reader:
            unlabeled_addr.append(row[0])

    length = len(unlabeled_addr) // 49
    process_list = []

    for i in range(50):
        pid = Process(target=add_feature_to_list_addr, args=(unlabeled_addr[i * length: (i + 1) * length], i))
        process_list.append(pid)

    for pid in process_list:
        pid.start()

    for pid in process_list:
        pid.join()


# 将地址新提取的特征，加到cascade中的self_feature部分。
def concat_added_feature_to_removed():
    random_1_added_feature_path = "../addr_self_feature/random_addr_1_added_feature.csv"
    random_1_added_reader = csv.reader(open(random_1_added_feature_path, "r"))
    added_feature = dict()
    for row in random_1_added_reader:
        added_feature[row[0]] = row[1:]

    origin_unlabel_path = "../addr_cascade_feature/random_addr_1_cascade_feature_filtered_remove_amount.csv"
    origin_reader = csv.reader(open(origin_unlabel_path, "r"))

    save_path = "../addr_cascade_feature/random_1_concat.csv"
    writer = csv.writer(open(save_path, "a+"))
    for row in origin_reader:
        added_features = added_feature[row[0]]
        for index, feature in enumerate(added_features):
            row.insert(index + 2, feature)
        writer.writerow(row)

    ransomware_added_feature_path = "../addr_self_feature/rs_addr_added_features.csv"
    ransomware_added_reader = csv.reader(open(ransomware_added_feature_path, "r"))
    added_feature = dict()
    for row in ransomware_added_reader:
        added_feature[row[0]] = row[1:]

    origin_ransomware_path = "../addr_cascade_feature/rs_addr_clustered_remove_amount.csv"
    origin_reader = csv.reader(open(origin_ransomware_path, "r"))

    save_path = "../addr_cascade_feature/rs_addr_clustered_concat.csv"
    writer = csv.writer(open(save_path, "a+"))

    for row in origin_reader:
        added_features = added_feature[row[0]]
        for index, feature in enumerate(added_features):
            row.insert(index + 3, feature)
        writer.writerow(row)


# 使用麦可之前跑全量地址检测得到的交易对手地址构建rocksdb数据库
def construct_addr_oppos_db_using_old_data():
    addr_oppos_db_path = "../addr_oppos/addr_oppos_old.db"
    addr_oppos_db = rocksdb.DB(addr_oppos_db_path, rocksdb.Options(create_if_missing=True))
    folder = "../result_csv/"
    role_list = ["in_neighbor.csv", "out_neighbor.csv", "out_sib.csv", "in_sib.csv"]
    for year in range(2014, 2022):
        logger.info(year)
        year_path = folder + str(year) + "/"
        for index in range(10):
            logger.info(index)
            addr_oppos = dict()
            index_folder = year_path + str(index) + "/"
            for role in role_list:
                logger.info(role)
                role_path = index_folder + role
                reader = csv.reader(open(role_path, "r"), delimiter=';')
                count = 0
                for row in reader:
                    if count % 10000000 == 0:
                        logger.info(count)
                    count += 1
                    addr = row[0]
                    if addr not in addr_oppos:
                        addr_oppos[addr] = list()
                    addr_oppos[addr].append(row[1])

            for addr, oppos in addr_oppos.items():
                addr_oppos_db.put(bytes(addr, encoding='utf-8'), bytes(str(oppos), encoding='utf-8'))
                # logger.info(addr)
                # logger.info(oppos)
                # exit(0)

def extract_oppo_addr_extra_one_process(pid):


    save_folder = "../addr_self_feature/oppo_addr_extra_feature/"
    save_path = save_folder + str(pid) + ".csv"

    split_file_list = sorted(os.listdir("../addr_oppos/unip_oppo_addrs_split/"))

    writer = csv.writer(open(save_path, "a+"))
    addr_list = list()
    reader = csv.reader(open("../addr_oppos/unip_oppo_addrs_split/" + split_file_list[pid], "r"))
    for row in reader:
        addr_list.append(row[0])
    for addr in tqdm(addr_list):
        features = []
        addr_object = blockchain.address_from_string(addr)
        if addr_object is None:
            continue
        
        features.append(addr)
        features.append(addr_object.raw_type)
        max_zero_count = 0
        min_zero_count = sys.maxsize
        for value in list(addr_object.outputs.value):
            zero_count = 0
            for i in str(value)[::-1]:
                if i == '0':
                    zero_count += 1
                else:
                    break
            if zero_count > max_zero_count:
                max_zero_count = zero_count
            if zero_count < min_zero_count:
                min_zero_count = zero_count

        features.append(max_zero_count)
        features.append(min_zero_count)

        self_change = 0
        for tx in addr_object.txes.to_list():
            input_addrs = tx.inputs.address.to_list()
            output_addrs = tx.outputs.address.to_list()
            if addr_object in input_addrs and addr_object in output_addrs:
                self_change = 1
                break
        if self_change:
            features.append(1)
        else:
            features.append(0)
        
        writer.writerow(features)




# 提取random_1 random_2 对手地址的新特征, type, self_change
def extract_oppo_addr_extra_mul_process():
    pid_list = []
    for i in range(50):
        pid_list.append(Process(target=extract_oppo_addr_extra_one_process, args=(i, )))
    
    for p in pid_list:
        p.start()
    
    for p in pid_list:
        p.join()

def load_all_oppo_addr():
    all_oppo_addr = []
    path_list = ["../addr_oppos/random_addr_1_oppos_combined.csv", 
    "../addr_oppos/random_addr_2_oppos_combined.csv", 
    "../addr_oppos/random_entity_addr_oppos_combined.csv"]
    for path in path_list:
        reader = csv.reader(open(path, "r"))
        for row in reader:
            all_oppo_addr.append(row[0])
    
    return set(all_oppo_addr)



# 加载已做完的oppo_addr_num
def load_done_oppo_addr_num():
    folder = "../addr_oppos/oppo_addr_oppo_num/"
    done_addrs = []
    for year in range(2014,2022):
        for i in range(10):
            reader = csv.reader(open(folder + str(year) + "/" + str(i) + ".csv", "r"))
            for row in reader:
                done_addrs.append(row[0])
    
    return set(done_addrs)

# 提取对手地址的额外的特征
def extract_other_oppos_extra_feature():
    all_oppo_addr = load_all_oppo_addr()
    done_oppo_addr = load_done_oppo_addr_num()
    to_do_oppo_addr = list(all_oppo_addr - done_oppo_addr)
    logger.info(len(to_do_oppo_addr))

    process_length = len(to_do_oppo_addr) // 49

    for i in range(50):
        pid_list.append(Process(target=extract_oppo_addr_extra_one_process, args=(i, )))
    
    for p in pid_list:
        p.start()
    
    for p in pid_list:
        p.join()
        
    

    


# 统计全量赎金
def statistic_all_ransom():
    path = "all_ransom.csv"
    reader = csv.reader(open(path, "r"), delimiter=' ')
    count = 0
    for row in reader:
        count += eval(row[6].replace(',', ''))
    
    logger.info(count)


if __name__ == '__main__':
    # statistic_family_addr_clustered()
    # filter_ransomware_address_more_10()
    # add_features_to_addr("1JKqWuTFimZ2YYhb8w61wfHMeNpVXhqkYe")
    start_time = time.time()
    # add_feature_to_ransomware_addr()
    # add_feature_to_random_1_addr()
    # remove_features()
    # concat_added_feature_to_removed()
    # logger.info("test")
    # extract_oppo_addr_extra_mul_process()
    # extract_oppo_addr_extra_mul_process()
    extract_other_oppos_extra_feature()
    # test_blocksci()
    # construct_addr_oppos_db_using_old_data()
    end_time = time.time()
    logger.info(end_time - start_time)
    # logger.info("666")
    # combine_walletexplorer()
# get_address_family()
# filter_ransomware_cascade_feature()
# remove_features()


#   test_blocksci()
# add_extra_features_to_with_label()
# ransomware_address_path = "../nfs/wk/TDSC_major_revision/data/ransomware_dataset/final_ransomware_address.csv"
# reader = csv.reader(open(ransomware_address_path, "r"))
# count = 0
# for row in reader:
#     addr = row[0]
#     if check_in_and_out(addr):
#         count += 1

# logger.info(count)
