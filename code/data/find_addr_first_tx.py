import rocksdb
import os
import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger('find addr first tx date')

white_addr_file_loca = '../cd_data/cascade_feature_path/random_addr/combined_random_cascade_feature.csv'

ransom_addr_file_loca = '../nfs/zhaoyu/ransom_addr'


def find_addr_first_tx_date(one_addr, db):
    db_res = db.get(bytes(one_addr, encoding='utf-8'))
    one_addr_date_list = []
    if db_res is not None:
        db_res_txs = str(db_res, encoding='utf-8')
        db_res_txs = eval(fix_data_format(db_res_txs))
        for one_tx in db_res_txs:
            date = one_tx.split('/')[0]
            one_addr_date_list.append(date)
        first_date = min(one_addr_date_list)
        last_date = max(one_addr_date_list)
        return first_date, last_date
    else:
        return '0000s', '9999s'


def fix_data_format(data):
    data = data.replace('[', '')
    data = data.replace(']', '')
    data = '[' + data + ']'
    return data


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


if __name__ == '__main__':
    # 160数据
    white_addr_list, ransom_addr_list = load_white_addr_and_ransom_addr()
    logger.info('addr load finish...')
    db = rocksdb.DB('../cd_data/addr_related_txs.db', rocksdb.Options(create_if_missing=True))
    with open('../cd_data/white_addr_first_date.csv', 'a') as f1:
        cnt = 0
        for one_addr in white_addr_list:
            cnt += 1
            if cnt % 1000 == 0:
                logger.info(cnt)
            f_date, l_date = find_addr_first_tx_date(one_addr, db)
            f1.write(one_addr + ',' + f_date + ',' + l_date + '\n')
    f1.close()
    logger.info('white addr finish...')
    with open('../cd_data/ransom_addr_first_date.csv', 'a') as f2:
        cnt = 0
        for one_addr in ransom_addr_list:
            cnt += 1
            if cnt % 1000 == 0:
                logger.info(cnt)
            f_date, l_date = find_addr_first_tx_date(one_addr, db)
            f2.write(one_addr + ',' + f_date + ',' + l_date + '\n')
    f2.close()
    logger.info('ransoom addr finish...')
    del db
