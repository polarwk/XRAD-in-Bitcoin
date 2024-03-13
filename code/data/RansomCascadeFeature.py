import os, sys
import csv
from loguru import logger
from numpy import sort
import rocksdb
import numpy as np
from multiprocessing import Process
import math

# logger.add('RansomCascadeFeature.log', format="{time} {level} {message}", level="DEBUG", rotation="500 MB")

addr_feature_db_path = "../cd_data/addr_feature_new.db"
addr_opposite_loca_path = "../nfs/wk/TDSC/ransom_address_oppos/"
cascade_feature_path = "../nfs/wk/TDSC/addr_cascade_feature/ransom_new_sorted/"

Addr_Feature_DB = rocksdb.DB(addr_feature_db_path, rocksdb.Options(create_if_missing=True), read_only=True)

def take_length(elem):
    return len(elem)

def calculate_cascade_feature_each():
    pid_cascade_feature_path = cascade_feature_path + "ransom_cascade_feature.csv"
    csv_writer = csv.writer(open(pid_cascade_feature_path, "a+"))
    file_name_list = ["in_neighbor.csv", "out_neighbor.csv", "in_sib.csv", "out_sib.csv"]
    file_open_list = []
    for file in file_name_list:
        file_open_list.append(open(addr_opposite_loca_path + file))
    cnt = 0
    write_list = []
    while True:
        if cnt % 1000 == 0:
            csv_writer.writerows(write_list)
            write_list = []
            logger.debug("            ///////           " + str(cnt))
        cnt += 1

        opposite_addr_list = []
        in_neighbor_line = file_open_list[0].readline()
        out_neighbor_line = file_open_list[1].readline()
        in_sib_line = file_open_list[2].readline()
        out_sib_line = file_open_list[3].readline()
        if in_neighbor_line == "":
            break
        address = in_neighbor_line.split(";")[0]
        
        if address == "00000000":
            continue
        try:
            in_neighbors = eval(in_neighbor_line.split(";")[1])
            in_neighbor_length = len(in_neighbors)

            # in_neighbors.append(address)

            out_neighbors = eval(out_neighbor_line.split(";")[1])
            out_neighbor_length = len(out_neighbors)
            
            # out_neighbors.append(address)

            in_sibs = eval(in_sib_line.split(";")[1])
            in_sib_length = len(in_sibs)
            # in_sibs.append(address)

            out_sibs = eval(out_sib_line.split(";")[1])
            out_sib_length = len(out_sibs)
            # out_sibs.append(address)

            opposite_addr_list.append(in_neighbors)
            opposite_addr_list.append(out_neighbors)
            opposite_addr_list.append(in_sibs)
            opposite_addr_list.append(out_sibs)
            opposite_addr_list.sort(key=take_length, reverse = True)
        except Exception as ex:
          logger.info(ex)
          logger.info(out_neighbor_line)
        
        feature_itself = Addr_Feature_DB.get(bytes(address, encoding='utf-8'))
        if feature_itself is None:
            logger.info(":self:" + address)
            continue
        feature_itself = str(feature_itself, encoding='utf-8')

        feature_itself = feature_itself.replace("inf", "0")
        feature_itself = eval(feature_itself)

        features_length = len(feature_itself)

        for role_address_list in opposite_addr_list:
            role_address_list = role_address_list
            feature_list = []
            for opposite_address in role_address_list:
                if opposite_address == "00000000":
                    continue
                features_result = Addr_Feature_DB.get(bytes(opposite_address, encoding='utf-8'))
                if features_result is None:
                    logger.info(":oppo:" + opposite_address)
                

                features_result = str(features_result, encoding='utf-8')

                features = features_result.replace("inf", "0.0")
                features = eval(features)
                feature_list.append(features)
            if len(feature_list) == 0:
                for i in range(features_length):
                    if i == 15 or i == 16:
                        feature_itself.append(0)
                    else:
                        for j in range(4):
                            feature_itself.append(0.0)
                continue
            for i in range(features_length):
                # pass
                one_features_list = [float(one_features[i]) for one_features in feature_list]

                if i == 15 or i == 16:
                    feature_itself.append(max(one_features_list))
                else:
                    # 最大值
                    feature_itself.append(max(one_features_list))
                    # 最小值
                    feature_itself.append(min(one_features_list))
                    # 平均值
                    feature_itself.append(np.mean(one_features_list))
                    # 标准差
                    feature_itself.append(np.std(one_features_list))
        feature_itself.insert(0, address)
        write_list.append(feature_itself)
    csv_writer.writerows(write_list)





if __name__ == "__main__":

    calculate_cascade_feature_each()

    # process_list = []
    # for year in range(2013, 2021):
    #     process_x = Process(target=calculate_cascade_feature_each, args=(str(year), str(year)))
    #     process_list.append(process_x)
    
    # for process_x in process_list:
    #     process_x.start()
    
    # for process_x in process_list:
    #     process_x.join()
