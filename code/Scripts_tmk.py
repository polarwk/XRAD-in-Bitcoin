import os
import sys
import csv
from loguru import logger
# import blocksci
# import rocksdb
import requests
import time

csv.field_size_limit(sys.maxsize)
from tqdm import tqdm

# blockchain = blocksci.Blockchain("../BlockSci/config_2.json")


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
    logger.info(addr_object)
    # exit(0)
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

    origin_cascade_features_path_list = ["../addr_cascade_feature/random_addr_1_cascade_feature_filtered.csv",
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

    origin_cascade_features_path_list = ["../addr_cascade_feature/ransomware_addr_cascade_feature_with_label.csv"]
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


#过滤掉addr_value中的交易所地址
def filter_exchange():
    true_addr_set=set()
    writer = csv.writer(open("../nfs/wk/TDSC_major_revision/code/ransom_addr.csv", "w+"))
    reader = csv.reader(open("../ransomware_dataset/rs_addr_clustered.csv", "r"))
    for row in reader:
        true_addr_set.add(row[0])
    reader = csv.reader(open("../nfs/wk/TDSC_major_revision/code/addr_value.csv", "r"))
    for row in reader:
        if(len(row)==1):
            writer.writerow(row)
        elif(len(row)>1 and row[0] in true_addr_set):
            writer.writerow(row)

#根据金额过滤掉交易所地址
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

#获取地址所属勒索家族映射集合
def get_address_family():
    addr_family={}
    tag='7456'
    reader = csv.reader(open("../nfs/wk/TDSC_major_revision/code/extra_addr_value.csv", "r"))
    for row in reader:
        if(len(row)==1 and len(row[0])<20):
            tag=row[0]
        else:
            addr_family[row[0]]=tag
    
   # logger.info(addr_family['1PeFqJwKUWmdM225Gv8MnjhbzbXShzy9Kg'])
    return addr_family


#根据walletExplorer标签过滤非勒索地址的地址
def filter_with_we():
    addr_dict=get_address_family()
    result_list=[]
    not_address_set=set()
    reader = csv.reader(open("../ransomware_dataset/wrong_labeled_rs_addr.csv", "r"))
    writer = csv.writer(open("../nfs/wk/TDSC_major_revision/code/suspect_ransom_addr.csv", "w+"))
    for row in reader:
        not_address_set.add(row[0])
    



    reader = csv.reader(open("../nfs/wk/TDSC_major_revision/code/addr_value.csv", "r"))
    for row in reader:
        if(len(row)==1):
            writer.writerow(row)
        elif(len(row)>1 and row[0] not in not_address_set):
            row.append(addr_dict[row[0]])
            writer.writerow(row)

    


def filter_with_api_160():
    filepath='../ransomware_dataset/final_rs_addresses_single_label.csv'
    reader = csv.reader(open(filepath, "r"))
    
    write_path = "../nfs/wk/TDSC_major_revision/code/api_address-160.csv"

    addresses = list()
    for row in reader:
      addresses.append(row)

    addresses = addresses[-5000:]

    for row in tqdm(addresses):
        url="http://www.walletexplorer.com/api/1/address?address="+row[0]+"&from=0&count=100&caller=20110240065@fudan.edu.cn"
        tx_req = requests.get(url=url)
        result=eval(tx_req.text.replace("true","True").replace("false","False"))
        if('label' in result):
            f = open(write_path, "a+")
            writer = csv.writer(f)
            row.append(result['label'])
            writer.writerow(row)
            f.close()
        time.sleep(1.5)

def filter_with_api_70():
    filepath='../ransomware_dataset/final_rs_addresses_single_label.csv'
    reader = csv.reader(open(filepath, "r"))
    
    write_path = "../nfs/wk/TDSC_major_revision/code/api_address-70.csv"

    addresses = list()
    for row in reader:
      addresses.append(row)

    addresses = addresses[-10000:-5000]

    for row in tqdm(addresses):
        url="http://www.walletexplorer.com/api/1/address?address="+row[0]+"&from=0&count=100&caller=20110240077@fudan.edu.cn"
        tx_req = requests.get(url=url)
        result=eval(tx_req.text.replace("true","True").replace("false","False"))
        if('label' in result):
            f = open(write_path, "a+")
            writer = csv.writer(f)
            row.append(result['label'])
            writer.writerow(row)
            f.close()
        time.sleep(2)

def filter_with_api_55():
    filepath='../nfs/wk/TDSC_major_revision/data/ransomware_dataset/final_rs_addresses_single_label.csv'
    reader = csv.reader(open(filepath, "r"))
    
    write_path = "../nfs/wk/TDSC_major_revision/code/api_address-55.csv"

    addresses = list()
    for row in reader:
      addresses.append(row)

    addresses = addresses[-15000:-10000]

    for row in tqdm(addresses):
        url="http://www.walletexplorer.com/api/1/address?address="+row[0]+"&from=0&count=100&caller=203764526734@fudan.edu.cn"
        tx_req = requests.get(url=url)
        result=eval(tx_req.text.replace("true","True").replace("false","False"))
        if('label' in result):
            f = open(write_path, "a+")
            writer = csv.writer(f)
            row.append(result['label'])
            writer.writerow(row)
            f.close()
        time.sleep(2)

def filter_with_api_56():
    filepath='../nfs/wk/TDSC_major_revision/data/ransomware_dataset/final_rs_addresses_single_label.csv'
    reader = csv.reader(open(filepath, "r"))
    
    write_path = "../nfs/wk/TDSC_major_revision/code/api_address-56.csv"

    addresses = list()
    for row in reader:
      addresses.append(row)

    addresses = addresses[-20000:-15000]

    for row in tqdm(addresses):
        url="http://www.walletexplorer.com/api/1/address?address="+row[0]+"&from=0&count=100&caller=34534634243@fudan.edu.cn"
        tx_req = requests.get(url=url)
        result=eval(tx_req.text.replace("true","True").replace("false","False"))
        if('label' in result):
            f = open(write_path, "a+")
            writer = csv.writer(f)
            row.append(result['label'])
            writer.writerow(row)
            f.close()
        time.sleep(2)
      

#    
def filter_with_api():
    
    f=open("../nfs/wk/TDSC_major_revision/code/api_address.csv", "a+")
    writer = csv.writer(f)


    logger.info("start")
    count=0


    filepath='../ransomware_dataset/final_rs_addresses_single_label.csv'
    reader = csv.reader(open(filepath, "r"))

    for row in reader:
      address=row[0]
      if(count< 12750):
        count=count+1
        continue

      url="http://www.walletexplorer.com/api/1/address?address="+address+"&from=0&count=100&caller=20110240034@fudan.edu.cn"
      tx_req = requests.get(url=url)
      result=eval(tx_req.text.replace("true","True").replace("false","False"))
      if('label' in result):
        row.append(result['label'])
        writer.writerow(row)
      count=count+1
      time.sleep(4)
      if(count%50==0):
        logger.info(count)
        f.close()
        f=open("../nfs/wk/TDSC_major_revision/code/api_address.csv", "a+")
        writer = csv.writer(f)


def compare():
    CryptoLocker_set=set()
    filepath='../ransomware_dataset/padua/knowledge_base/addresses/CryptoLocker_addresses.txt'
    reader = csv.reader(open(filepath, "r"))
    for row in reader:
        CryptoLocker_set.add(row[0])
    
    CryptoWall_set=set()
    filepath='../ransomware_dataset/padua/knowledge_base/addresses/CryptoWall_addresses.txt'
    reader = csv.reader(open(filepath, "r"))
    for row in reader:
        CryptoWall_set.add(row[0])
    
    wrong_set=set()
    filepath='../ransomware_dataset/wrong_labeled_rs_addr.csv'
    reader = csv.reader(open(filepath, "r"))
    for row in reader:
        wrong_set.add(row[0])
    
    logger.info(len(CryptoLocker_set-wrong_set))

    logger.info(len(CryptoWall_set-wrong_set))


#探索勒索地址规律
def find_ransomware_rules(): 
    addr_dict=get_address_family()
    filepath='../nfs/wk/TDSC_major_revision/code/extra_addr_value.csv'
    reader = csv.reader(open(filepath, "r"))
    ransom_family_dict={}
    ransom_family_count={}
    for row in reader:
        for i in range(1,len(row)-1):
                value=row[i]
                family=addr_dict[row[0]]
                if(family not in ransom_family_dict):
                    ransom_family_dict[family]=[0,0]
                    ransom_family_count[family]=0
                
                ransom_family_count[family]=ransom_family_count[family]+1


                if(value[-1]!='0'):
                    ransom_family_dict[family][1]=ransom_family_dict[family][1]+1
                else:
                    ransom_family_dict[family][0]=ransom_family_dict[family][0]+1
    
    for key in ransom_family_dict.keys():
        total=ransom_family_dict[key][0]+ransom_family_dict[key][1]
        logger.info(key+"-------------"+str(ransom_family_dict[key][0]/total))
        logger.info(ransom_family_count[key])
        logger.info("****************************")


        
#每个活动划分多个金额区间
def find_ransom_section():
   # f=open("../nfs/wk/TDSC_major_revision/code/small.csv", "w+")
    # writer = csv.writer(f)
    addr_family={}
    basic_value=1e8
    tag='7456'
    addr_set=set()
    reader = csv.reader(open("../nfs/wk/TDSC_major_revision/code/ransom_addr.csv", "r"))
    for row in reader:
        if(len(row)==1 and len(row[0])<20):
            tag=row[0]
            if tag not in addr_family:
                addr_family[tag]=30*[0]
        else:
            if(len(row)>1):
                for i in range(1,len(row)):
                    temp_value=int(row[i])
                    result=temp_value/basic_value
                    if(result >=3):
                        addr_family[tag][29]=addr_family[tag][29]+1
            #            addr_set.add(row[0])
                    else:
                        locate=int(result/0.1)
                        addr_family[tag][locate]=addr_family[tag][locate]+1
    

    for key in addr_family.keys():
        logger.info(key+"----------"+str(addr_family[key]))
              
    return addr_family

#每个活动划分多个金额区间
def find_ransom_section_small():
   # f=open("../nfs/wk/TDSC_major_revision/code/small.csv", "w+")
    # writer = csv.writer(f)
    addr_family={}
    basic_value=1e8
    tag='7456'
    addr_set=set()
    reader = csv.reader(open("../nfs/wk/TDSC_major_revision/code/ransom_addr.csv", "r"))
    for row in reader:
        if(len(row)==1 and len(row[0])<20):
            tag=row[0]
            if tag not in addr_family:
                addr_family[tag]=500*[0]
        else:
            if(len(row)>1):
                for i in range(1,len(row)):
                    temp_value=int(row[i])
                    result=temp_value/basic_value
                    if(result >= 5):
                        addr_family[tag][499]=addr_family[tag][499]+1
                    else:
                        locate=int(result/0.01)
                        addr_family[tag][locate]=addr_family[tag][locate]+1
    

    for key in addr_family.keys():
        logger.info(key+"----------"+str(addr_family[key]))
              
    return addr_family
 
if __name__ == '__main__':
   # filter_exchange()
   #  find_ransom_section()
     find_ransom_section_small()
  #   find_ransom_section()
    
  #  find_ransomware_rules()
    # combine_walletexplorer()
   # get_address_family()
  #  filter_with_api()

  #  filter_with_we()

 #   filter_with_heurstics()
   
   
   # compare()





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
