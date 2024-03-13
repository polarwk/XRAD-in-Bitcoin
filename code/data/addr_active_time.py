# 160_data
# 这份代码是根据地址，选取地址的活跃时间范围
import os
import time
import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger('set addr active time')

white_addr_file_loca = '../cd_data/cascade_feature_path/random_addr/combined_random_cascade_feature.csv'

ransom_addr_file_loca = '../nfs/zhaoyu/ransom_addr'

addr_tx_time_data_loca = '../addr_link/update_to_2021_12_31.csv'



def load_white_addr_and_ransom_addr():
    white_addr_list = []
    ransom_addr_list = []
    for line in open(white_addr_file_loca, 'r'):
        line = line.split(',')
        addr = line[0]
        white_addr_list.append(addr)

    all_ransoms = os.listdir(ransom_addr_file_loca)
    for one_ransom in all_ransoms:
        for line in open(ransom_addr_file_loca + '/' + one_ransom, 'r'):
            line = line.split(',')
            addr = line[0]
            ransom_addr_list.append(addr)
    return white_addr_list, ransom_addr_list


def set_addr_active_time(white_addr_list_data, ransom_addr_list_data):
    white_addr_list_data = set(white_addr_list_data)
    ransom_addr_list_data = set(ransom_addr_list_data)
    logger.info('start update_to_2021_12_31.csv')
    white_addr_time_dict = {}
    ransom_addr_time_dict = {}
    cnt = 0
    for line in open(addr_tx_time_data_loca, 'r'):
        cnt += 1
        if cnt % 10000 == 0:
            logger.info(cnt)
        if '\n' in line:
            line = line[0:-1]
        line = line.split(';')
        addr = line[0]
        if addr in white_addr_list_data:
            first_time_stamp = eval(line[1])
            last_time_stamp = eval(line[3])
            first_time = time.localtime(first_time_stamp)
            last_time = time.localtime(last_time_stamp)
            first_time = time.strftime("%Y/%m/%d %H:%M:%S", first_time)
            last_time = time.strftime("%Y/%m/%d %H:%M:%S", last_time)
            white_addr_time_dict[addr] = [first_time, last_time]
        elif addr in ransom_addr_list_data:
            first_time_stamp = eval(line[1])
            last_time_stamp = eval(line[3])
            first_time = time.localtime(first_time_stamp)
            last_time = time.localtime(last_time_stamp)
            first_time = time.strftime("%Y/%m/%d %H:%M:%S", first_time)
            last_time = time.strftime("%Y/%m/%d %H:%M:%S", last_time)
            ransom_addr_time_dict[addr] = [first_time, last_time]
    return white_addr_time_dict, ransom_addr_time_dict


if __name__ == '__main__':
    white_addr_list, ransom_addr_list = load_white_addr_and_ransom_addr()
    logger.info('addr load finish...')
    white_addr_time_res, ransom_addr_time_res = set_addr_active_time(white_addr_list, ransom_addr_list)
    logger.info('addr time find finish...')

    logger.info('white_addr_time_res: ' + str(len(white_addr_time_res)))
    logger.info('ransom_addr_time_res: ' + str(len(ransom_addr_time_res)))

    with open('../cd_data/white_addr_active.csv', 'a') as f1:
        for item in white_addr_time_res:
            f1.write(item + ',' + white_addr_time_res[item][0] + ',' + white_addr_time_res[item][1] + '\n')
    f1.close()

    logger.info('white finish...')
    with open('../cd_data/ransom_addr_active.csv', 'a') as f2:
        for item in ransom_addr_time_res:
            f2.write(item + ',' + ransom_addr_time_res[item][0] + ',' + ransom_addr_time_res[item][1] + '\n')
    f2.close()
    logger.info('ransom finish...')
