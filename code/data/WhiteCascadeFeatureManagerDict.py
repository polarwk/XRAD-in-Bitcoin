import os, sys
import csv
from loguru import logger
import rocksdb
import numpy as np
from multiprocessing import Process
import multiprocessing as mp
import math

# logger.add('CascadeFeatureManagerDict.log', format="{time} {level} {message}", level="DEBUG", rotation="500 MB")

addr_feature_db_path = "../addr_feature_new.db"
addr_opposite_loca_path = "../nfs/wk/TDSC/white_address_oppos/"
cascade_feature_path = "../nfs/wk/TDSC/addr_cascade_feature/white_new_sorted/"

Addr_Feature_DB = rocksdb.DB(addr_feature_db_path, rocksdb.Options(create_if_missing=True), read_only=True)
# addr_feature_dict=mp.Manager().dict()

def take_length(elem):
    return len(elem)

def calculate_cascade_feature_each(process, pid):
    logger.debug(str(pid) + "   starting")
    pid_cascade_feature_path = cascade_feature_path + str(pid) + "_cascade_feature.csv"
    csv_writer = csv.writer(open(pid_cascade_feature_path, "a+"))
    file_name_list = ["in_neighbor.csv", "out_neighbor.csv", "in_sib.csv", "out_sib.csv"]
    file_open_list = []
    for file in file_name_list:
        file_open_list.append(open(addr_opposite_loca_path + process + "/" + file))
    cnt = 0
    write_list = []
    while True:
        if cnt % 10000 == 0:
            csv_writer.writerows(write_list)
            write_list = []
            logger.debug(str(pid) + "            ///////           " + str(cnt))
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
            logger.info(str(pid) + ":self:" + address)
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
                    logger.info(str(pid) + ":oppo:" + opposite_address)
                

                features_result = str(features_result, encoding='utf-8')

                features = features_result.replace("inf", "0.0")
                features = eval(features)
                feature_list.append(features)
            if len(feature_list) == 0:
                for i in range(features_length):
                    if i == 15 or i == 16:
                        feature_itself.append(0.0)
                    else:
                        for j in range(4):
                            feature_itself.append(0.0)
                continue
            for i in range(features_length):
                # pass
                one_features_list = [float(one_features[i]) for one_features in feature_list]
                # max_value, min_value, mean_value, std_value = compute_list_attribute(one_features_list)
                # feature_itself.append(max_value)
                # feature_itself.append(min_value)
                # feature_itself.append(mean_value)
                # feature_itself.append(std_value)
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
        # csv_writer.writerow(feature_itself)


def compute_list_attribute(list):
    max_value = list[0]
    min_value = list[0]
    mean_value = list[0]
    std_value = list[0]
    sum_value = 0
    for item in list:
        sum_value += item
        if item > max_value:
            max_value = item
        if item < min_value:
            min_value = item

    mean_value = sum_value / len(list)

    sqrt_value = 0
    for item in list:
        sqrt_value += (item - mean_value) * (item - mean_value)
    std_value = math.sqrt(sqrt_value)

    return max_value, min_value, mean_value, std_value


def load_addr_feature():
    # process_list = os.listdir(opposite_path)
    temp_addr_feature_dict = dict()
    cnt = 0
    csv_reader = csv.reader(open("../nfs/tmk/www_experiment/resource_csv/2018/2018_addr.csv", "r"))
    for row in csv_reader:
            if cnt % 10000 == 0:
                logger.debug(cnt)
            cnt += 1
            address = row[0]
            feature_result = Addr_Feature_DB.get(bytes(address, encoding='utf-8'))
            if feature_result is not None:
                feature_result = str(feature_result, encoding='utf-8')
                temp_addr_feature_dict[address] = feature_result
    csv_reader = csv.reader(open("../nfs/tmk/www_experiment/result_csv/2018/2018.csv", "r"))
    cnt = 0
    for row in csv_reader:
        if cnt % 10000 == 0:
            logger.error(cnt)
        cnt += 1
        address = row[0]
        feature_result = Addr_Feature_DB.get(bytes(address, encoding='utf-8'))
        if feature_result is not None:
            feature_result = str(feature_result, encoding='utf-8')
            temp_addr_feature_dict[address] = feature_result
    addr_feature_dict.update(temp_addr_feature_dict)
    logger.info("finish transfer")

if __name__ == "__main__":

    process_list = []
    for year in range(2013, 2021):
        process_x = Process(target=calculate_cascade_feature_each, args=(str(year), str(year)))
        process_list.append(process_x)
    
    for process_x in process_list:
        process_x.start()
    
    for process_x in process_list:
        process_x.join()


    



    #原版
  #   mp_dict = mp.Manager().dict()
  #   addr_feature_dict=load_addr_feature()
    #  load_addr_feature()
   #   2019
   #  process_list=['0','1','2','4','7','8','9']
    #  process_list=['0','1','2','7','8']
     
    #  process_x_list = []
    #  for i in range(len(process_list)):
    #      process_x = Process(target=calculate_cascade_feature_each, args=(process_list[i],process_list[i]))
    #      process_x_list.append(process_x)

    #  for p in process_x_list:
    #      p.daemon = True
    #      p.start()

    #  for p in process_x_list:
    #      p.join()


    #单进程补漏版
#    addr_feature_dict=load_addr_feature()
#    calculate_cascade_feature_each('3',3,addr_feature_dict)


    
# def calculate_cascade_feature():
#     addr_2016_6_30_path = "../www_db/year_address/2016_6_30_addr.csv"
#     csv_reader = csv.reader(open(addr_2016_6_30_path, "r"))
#     pid_cascade_feature_path = cascade_feature_path + "cascade_feature.csv"
#     csv_writer = csv.writer(open(pid_cascade_feature_path, "a+"))
#     cnt = 0
#     for row in csv_reader:
#         if cnt % 1000 == 0:
#             logger.debug(cnt)
#         cnt += 1
#         address = row[0]
#
#         opposite_address_list_result = Addr_Opposite_DB.get(bytes(address, encoding='utf-8'))
#         if opposite_address_list_result is None:
#             continue
#         opposite_address_list = str(opposite_address_list_result, encoding='utf-8')
#         opposite_address_list = eval(opposite_address_list)
#
#         feature_itself = str(Addr_Feature_DB.get(bytes(address, encoding='utf-8')), encoding='utf-8')
#         feature_itself = feature_itself.replace("inf", "0")
#         feature_itself = eval(feature_itself)
#         features_length = len(feature_itself)
#         for role_address_list in opposite_address_list:
#             role_address_list = eval(role_address_list)
#             feature_list = []
#             for opposite_address in role_address_list:
#                 features_result = Addr_Feature_DB.get(bytes(opposite_address, encoding='utf-8'))
#                 if features_result is None:
#                     logger.error("opposite addr feature miss")
#                     continue
#                 features_result = str(features_result, encoding='utf-8')
#                 features = features_result.replace("inf", "0.0")
#                 features = eval(features)
#                 feature_list.append(features)
#             if len(feature_list) == 0:
#                 for i in range(features_length):
#                     for j in range(4):
#                         feature_itself.append(0.0)
#                 continue
#             for i in range(features_length):
#                 one_features_list = [float(one_features[i]) for one_features in feature_list]
#
#                 # 最大值
#                 feature_itself.append(max(one_features_list))
#                 # 最小值
#                 feature_itself.append(min(one_features_list))
#                 # 平均值
#                 feature_itself.append(np.mean(one_features_list))
#                 # 标准差
#                 feature_itself.append(np.std(one_features_list))
#         feature_itself.insert(0, address)
#         csv_writer.writerow(feature_itself)
