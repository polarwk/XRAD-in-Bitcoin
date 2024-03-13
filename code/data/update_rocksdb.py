import rocksdb
import logging
import os
import json

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("insert DB ")

tx_data_loca = '../bitcoin_data_local_file/2021/'


# 将交易信息更新至rocksdb
def write_related_tx_into_db(date):
    # 这里是rocksdb的路径
    db = rocksdb.DB('../addr_related_txs.db', rocksdb.Options(create_if_missing=True))
    one_day_res = get_addr_all_txs(date)
    for one_addr in one_day_res:
        db_res = db.get(bytes(one_addr, encoding='utf-8'))
        if db_res is None:
            need_to_write = list(set(one_day_res[one_addr]))
            db.put(bytes(one_addr, encoding='utf-8'), bytes(str(need_to_write), encoding='utf-8'))
        else:
            db_res_txs = eval(str(db_res, encoding='utf-8'))
            db_res_txs = fix_data_format(db_res_txs)
            db_res_txs.extend(one_day_res[one_addr])
            need_to_write = list(set(one_day_res[one_addr]))
            db.put(bytes(one_addr, encoding='utf-8'), bytes(str(need_to_write), encoding='utf-8'))


# 修补原先地址相关交易里的 ‘[’ ']'符号bug问题
def fix_data_format(data):
    data = str(data)
    data = data.replace('[', '')
    data = data.replace(']', '')
    data = '[' + data + ']'
    return list(data)


# 遍历1天的交易，得到1天内涉及的地址相关交易数据
def get_addr_all_txs(date):
    address_tx_tmp = {}
    logger.info(date + 'starting...')
    one_day_all_jsons = os.listdir(tx_data_loca + '/' + date)
    one_day_all_jsons.sort()
    for each_json in one_day_all_jsons:
        #   filename='1609545500_664061.json'
        block_height = each_json.split(".")[0].split("_")[1]
        for line in open(tx_data_loca + '/' + date + '/' + each_json, 'r'):
            line_output = line
            line = json.loads(line)
            vin = line.get('vin')
            vout = line.get('vout')
            if vin is not None:
                try:
                    for vin_item in vin:
                        if vin_item.get('address') is not None \
                                and vin_item.get('address') != '0000000000000000000000000000000000':
                            if vin_item.get('address') not in address_tx_tmp:
                                address_tx_tmp[vin_item.get('address')] = [
                                    date + '/' + block_height + '/' + line['txhash']]
                            else:
                                address_tx_tmp[vin_item.get('address')] += [
                                    date + '/' + block_height + '/' + line['txhash']]
                except Exception as e:
                    logger.info(line_output)
            if vout is not None:
                for vout_item in vout:
                    if vout_item.get('address') is not None \
                            and vout_item.get('address') != '0000000000000000000000000000000000':
                        if vout_item.get('address') not in address_tx_tmp:
                            address_tx_tmp[vout_item.get('address')] = [
                                date + '/' + block_height + '/' + line['txhash']]
                        else:
                            address_tx_tmp[vout_item.get('address')] += [
                                date + '/' + block_height + '/' + line['txhash']]
        logger.info(each_json + "处理完毕")
    return address_tx_tmp


def get_one_year_addr(year):
    year_tx_loca = '../bitcoin_data_local_file/' + str(year)
    address_list = []
    date_list = os.listdir(year_tx_loca)
    date_list.sort()
    for date in date_list:
        one_day_all_jsons = os.listdir(year_tx_loca + '/' + date)
        one_day_all_jsons.sort()
        for each_json in one_day_all_jsons:
            #   filename='1609545500_664061.json'
            block_height = each_json.split(".")[0].split("_")[1]
            for line in open(year_tx_loca + '/' + date + '/' + each_json, 'r'):
                line_output = line
                line = json.loads(line)
                vin = line.get('vin')
                vout = line.get('vout')
                if vin is not None:
                    try:
                        for vin_item in vin:
                            if vin_item.get('address') is not None \
                                    and vin_item.get('address') != '0000000000000000000000000000000000':
                                address_list.append(vin_item.get('address'))
                    except Exception as e:
                        logger.info(line_output)
                if vout is not None:
                    for vout_item in vout:
                        if vout_item.get('address') is not None \
                                and vout_item.get('address') != '0000000000000000000000000000000000':
                            address_list.append(vout_item.get('address'))
            logger.info(each_json + "处理完毕")
        return list(set(address_list))


# 这个方法最好再确认一下，保持和原有的数据格式一致，这里我记不清了
def write_one_year_addr(data):
    with open('这里添加写文件的路径', 'a') as f1:
        for one_addr in data:
            f1.write(one_addr + '\n')
    f1.close()


if __name__ == '__main__':
    date_list = os.listdir(tx_data_loca)
    date_list.sort()
    #  logger.info(date_list)
    #  exit(0)
    for date in date_list:
        write_related_tx_into_db(date)
#  break
