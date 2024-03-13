import os
import sys
import csv
from loguru import logger
from tqdm import tqdm
from typing import Set, List
from multiprocess import Process
import blocksci
import rocksdb
import time

csv.field_size_limit(sys.maxsize)

blockchain = blocksci.Blockchain("../BlockSci-0.7.0/config.json")
addr_oppos_path = "../addr_oppos/addr_oppos.db"


def get_all_mixTransactions() -> Set[str]:
    path = "../mixingTx.csv"
    reader = csv.reader(open(path, "r"))
    mixTransactions = set()
    for row in reader:
        mixTransactions.add(row[0])

    return mixTransactions


def fing_addr_oppos(addr: str):
    in_neighbor = []
    out_neighbor = []
    out_sibling = []
    in_sibling = []
    # dict结构：{'addr':[in_neighbor, out_neighbor, in_sib, out_sib]}
    addr_info_dict = {}

    addr_object = blockchain.address_from_string(addr)
    if addr_object is None:
        return [in_neighbor, out_neighbor, out_sibling, in_sibling]

    # for tx in addr_object.txes:
    #     if str(tx.hash) in mixTransactions:
    #         continue
    As_input_tx = addr_object.input_txes.to_list()
    As_output_tx = addr_object.output_txes.to_list()
    # for tx in addr_object.input_txes.to_list():
    #     As_input_tx.append(tx)

    # for tx in addr_object.output_txes:
    #     As_output_tx.append(tx)
    
    As_input_tx = set(As_input_tx)
    As_output_tx = set(As_output_tx)


    overlap = As_input_tx & As_output_tx

    # 当该地址作为input时
    for tx in As_input_tx - overlap:
        if str(tx.hash) in mixTransactions:
            continue
        for input_address in tx.inputs.address.to_list():
            # input_address = input.address
            if type(input_address) is blocksci.OpReturn or type(input_address) is blocksci.NonStandardAddress or type(input_address) is blocksci.WitnessUnknownAddress:
                continue
            if type(input_address) is blocksci.MultisigAddress:
                in_addr = input_address.addresses[0].address_string
            else:
                in_addr = input_address.address_string

            if in_addr == addr:
                continue

            if in_addr not in addr_info_dict:
                addr_info_dict[in_addr] = [0, 0, 1, 0]
            else:
                tmp_val = addr_info_dict[in_addr]
                tmp_val[2] = tmp_val[2] + 1
                addr_info_dict[in_addr] = tmp_val

        for output_address in tx.outputs.address.to_list():
            # output_address = output.address
            if type(output_address) is blocksci.OpReturn or type(output_address) is blocksci.NonStandardAddress or type(output_address) is blocksci.WitnessUnknownAddress:
                continue
            if type(output_address) is blocksci.MultisigAddress:
                out_addr = output_address.addresses[0].address_string
            else:
                out_addr = output_address.address_string

            if out_addr == addr:
                continue

            if out_addr not in addr_info_dict:
                addr_info_dict[out_addr] = [0, 1, 0, 0]
            else:
                tmp_val = addr_info_dict[out_addr]
                tmp_val[1] = tmp_val[1] + 1
                addr_info_dict[out_addr] = tmp_val

    # 当该地址作为output时
    for tx in As_output_tx - overlap:
        if str(tx.hash) in mixTransactions:
            continue
        for input_address in tx.inputs.address.to_list():
            # input_address = input.address
            if type(input_address) is blocksci.OpReturn or type(input_address) is blocksci.NonStandardAddress or type(input_address) is blocksci.WitnessUnknownAddress:
                continue
            if type(input_address) is blocksci.MultisigAddress:
                in_addr = input_address.addresses[0].address_string
            else:
                in_addr = input_address.address_string

            if in_addr == addr:
                continue

            if in_addr not in addr_info_dict:
                addr_info_dict[in_addr] = [1, 0, 0, 0]
            else:
                tmp_val = addr_info_dict[in_addr]
                tmp_val[0] = tmp_val[0] + 1
                addr_info_dict[in_addr] = tmp_val

        for output_address in tx.outputs.address.to_list():
            # output_address = output.address
            if type(output_address) is blocksci.OpReturn or type(output_address) is blocksci.NonStandardAddress or type(output_address) is blocksci.WitnessUnknownAddress:
                continue
            if type(output_address) is blocksci.MultisigAddress:
                out_addr = output_address.addresses[0].address_string
            else:
                # try:
                out_addr = output_address.address_string
                # except:
                #     logger.info(str(tx.hash))
                #     logger.info(output.address.balance)
                #     exit(0)
            if out_addr == addr:
                continue

            if out_addr not in addr_info_dict:
                addr_info_dict[out_addr] = [0, 0, 0, 1]
            else:
                tmp_val = addr_info_dict[out_addr]
                tmp_val[3] = tmp_val[3] + 1
                addr_info_dict[out_addr] = tmp_val

    # 当地址即在交易输入也在交易输出时
    for tx in overlap:
        if str(tx.hash) in mixTransactions:
            continue
        for input_address in tx.inputs.address.to_list():
            # input_address = input.address
            if type(input_address) is blocksci.OpReturn or type(input_address) is blocksci.NonStandardAddress or type(input_address) is blocksci.WitnessUnknownAddress:
                continue
            if type(input_address) is blocksci.MultisigAddress:
                in_addr = input_address.addresses[0].address_string
            else:
                in_addr = input_address.address_string

            if in_addr == addr:
                continue

            if in_addr not in addr_info_dict:
                addr_info_dict[in_addr] = [0, 0, 1, 0]
            else:
                tmp_val = addr_info_dict[in_addr]
                tmp_val[2] = tmp_val[2] + 1
                addr_info_dict[in_addr] = tmp_val

        for output_address in tx.outputs.address.to_list():
            # output_address = output.address
            if type(output_address) is blocksci.OpReturn or type(output_address) is blocksci.NonStandardAddress or type(output_address) is blocksci.WitnessUnknownAddress:
                continue
            if type(output_address) is blocksci.MultisigAddress:
                out_addr = output_address.addresses[0].address_string
            else:
                out_addr = output_address.address_string

            if out_addr == addr:
                continue

            if out_addr not in addr_info_dict:
                addr_info_dict[out_addr] = [0, 1, 0, 0]
            else:
                tmp_val = addr_info_dict[out_addr]
                tmp_val[1] = tmp_val[1] + 1
                addr_info_dict[out_addr] = tmp_val
            

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


def extract_ransomware_addr_opps():
    ransomware_address_path = "../ransomware_dataset/final_ransomware_address.csv"
    reader = csv.reader(open(ransomware_address_path, "r"))

    writer = csv.writer(open("../addr_oppos/ransomware_addr_oppos.csv", "a+"), delimiter=";")
    count = 0
    for row in reader:
        if count % 100 == 0:
            logger.info(count)
        count += 1
        if count < 21700:
            continue 
        addr = row[0]
        oppos = fing_addr_oppos(addr)
        writer.writerow([addr, oppos])

def extract_ransomware_addr_opps_one_process(addr_list: List, pid: int):
    writer = csv.writer(open("../addr_oppos/ransomware_addr_oppos/" + str(pid) + ".csv", "a+"), delimiter=";")
    for addr in addr_list:
        oppos = fing_addr_oppos(addr)
        writer.writerow([addr, oppos])

def extract_ransomware_addr_opps_mul_process():
    ransomware_address_path = "../ransomware_dataset/final_ransomware_address.csv"
    reader = csv.reader(open(ransomware_address_path, "r"))
    ransomware_address = list()
    for row in reader:
        addr = row[0]
        ransomware_address.append(addr)
    
    one_process_length = len(ransomware_address) // 59
    pid_list = []
    for i in range(60):
        pid_list.append(Process(target=extract_ransomware_addr_opps_one_process, args=(ransomware_address[i*one_process_length: (i+1) * one_process_length],i, )))
    
    for p in pid_list:
        p.start()
    
    for p in pid_list:
        p.join()

def extract_random_white_one_process(addr_list: List, pid: int):
    writer = csv.writer(open("../addr_oppos/random_addr_2_oppos/" + str(pid) + ".csv", "a+"), delimiter=";")
    for addr in tqdm(addr_list):
        oppos = fing_addr_oppos(addr)
        writer.writerow([addr, oppos])

def extract_random_white_2_oppos_mul_process():
    ransomware_address_path = "../random_addr_1000000_2.csv"
    reader = csv.reader(open(ransomware_address_path, "r"))
    ransomware_address = list()
    for row in reader:
        addr = row[0]
        ransomware_address.append(addr)

    one_process_length = len(ransomware_address) // 59
    pid_list = []
    for i in range(60):
        pid_list.append(Process(target=extract_random_white_one_process, args=(ransomware_address[i*one_process_length: (i+1) * one_process_length],i, )))
    
    for p in pid_list:
        p.start()
    
    for p in pid_list:
        p.join()


def extract_random_entity_one_process(addr_list: List, pid: int):
    writer = csv.writer(open("../addr_oppos/random_entity_addr_oppos-160/" + str(pid) + ".csv", "a+"), delimiter=";")
    for addr in tqdm(addr_list):
        oppos = fing_addr_oppos(addr)
        writer.writerow([addr, oppos])

def extract_random_entity_addr_oppos_mul_process():
    ransomware_address_path = "../random_entity_addr.csv"
    reader = csv.reader(open(ransomware_address_path, "r"))
    ransomware_address = list()
    for row in reader:
        addr = row[0]
        ransomware_address.append(addr)
    
    ransomware_address = ransomware_address[700000:]

    one_process_length = len(ransomware_address) // 39
    pid_list = []
    for i in range(40):
        pid_list.append(Process(target=extract_random_entity_one_process, args=(ransomware_address[i*one_process_length: (i+1) * one_process_length],i, )))
    
    for p in pid_list:
        p.start()
    
    for p in pid_list:
        p.join()



def upload_addr_oppos_folder_rocksdb():
    addr_oppos_db = rocksdb.DB(addr_oppos_path, rocksdb.Options(create_if_missing=True))
    folder = "../addr_oppos/random_addr_2_oppos/"
    file_list = os.listdir(folder)
    for file in tqdm(file_list):
        file_path = folder + file
        reader = csv.reader(open(file_path, "r"), delimiter=";")
        for row in reader:
            addr = row[0]
            oppos = row[1]
        
            addr_oppos_db.put(bytes(addr, encoding='utf-8'), bytes(oppos, encoding='utf-8'))

def upload_addr_oppos_rocksdb():
    addr_oppos_db = rocksdb.DB(addr_oppos_path, rocksdb.Options(create_if_missing=True))
    reader = csv.reader(open("../addr_oppos/random_addr_1_oppos_combined.csv", "r"), delimiter=";")
    for row in reader:
            addr = row[0]
            oppos = row[1]
            addr_oppos_db.put(bytes(addr, encoding='utf-8'), bytes(oppos, encoding='utf-8'))


def merge_oppos(folder_path: str)-> Set:
    oppo_addrs = list()
    file_list = os.listdir(folder_path)
    for file in file_list:
        logger.info(file)
        file_path = folder_path + file
        reader = csv.reader(open(file_path, "r"), delimiter=";")
        for row in reader:
            addr = row[0]
            oppos = eval(row[1])
            for addr_list in oppos:
                for addr in addr_list:
                    oppo_addrs.append(addr)
    oppo_addrs = set(oppo_addrs)
    writer = csv.writer(open("../addr_oppos/random_addr_2_oppos_combined.csv", "a+"))
    for addr in oppo_addrs:
        writer.writerow([addr])
    return oppo_addrs



def test_blocksci():
    addr_object = blockchain.address_from_string("1GaVKrVT17DN4dnWbTqGB9qG3rQrk1JBe9")
    logger.info(addr_object.txes.to_list())
    # for block in tqdm(blockchain.blocks[300000:400000]):
    #     for tx in block.txes:
    #         logger.info(tx.inputs.address.to_list())
    #         logger.info(tx.outputs.address.to_list()[0].address_string)
    #         exit(0)
            # for input in tx.inputs:
            #     if type(input.address) is blocksci.WitnessUnknownAddress:
            #         logger.info(tx.hash)
            #         logger.info(input.address)
            
            # for output in tx.outputs:
            #     if type(output.address) is blocksci.WitnessUnknownAddress:
            #         logger.info(tx.hash)
            #         logger.info(output.address)

def find_one_addr_oppos_upload(addr: str):
    addr_oppos_db = rocksdb.DB(addr_oppos_path, rocksdb.Options(create_if_missing=True))
    oppos = str(fing_addr_oppos(addr))
    addr_oppos_db.put(bytes(addr, encoding='utf-8'), bytes(oppos, encoding='utf-8'))


if __name__ == "__main__":
    start_time = time.time()
    mixTransactions = get_all_mixTransactions()
    extract_random_entity_addr_oppos_mul_process()
    # find_one_addr_oppos_upload("12i8GBDDLx93AnCdsGQTdN46EcDAz39vXa")
    # logger.info("load mixing transaction finish")
    # extract_random_white_2_oppos_mul_process()
    # logger.info("finish")
    # test_blocksci()
    # merge_oppos("../addr_oppos/random_addr_2_oppos/")
    # upload_addr_oppos_folder_rocksdb()
    end_time = time.time()
    logger.info(end_time - start_time)
