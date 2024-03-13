import os
import sys
import csv
from loguru import logger
csv.field_size_limit(sys.maxsize)
import time

#利用55，56，57，58，59
def evaluate_completeness(date):
    basic_path="../nfs/bitcoin_mempool_data/microsecond/"
    ip_list=['55','56','57','58','59']
    target_list=['55']
    for ip in ip_list:
      filelist=os.listdir('../nfs/bitcoin_mempool_data/microsecond/'+ip+'/inv_ip/')
      last_list=[]
      filelist.sort()
      for i in range(len(filelist)):
        f=filelist[i]
        if(f.split("_")[0] in date):
            logger.info(f)
            last_list.append(i)
        #    last_list.append('../nfs/bitcoin_mempool_data/microsecond/'+ip+'/inv_ip/'+f)
      
      last_list.append(last_list[-1]+1)
      last_list.append(last_list[0]-1) 

      for index in last_list:
        csv_reader = csv.reader(open('../nfs/bitcoin_mempool_data/microsecond/'+ip+'/inv_ip/'+filelist[index], "r" , encoding='UTF-8-sig'), delimiter=',')
        for row in csv_reader:
            







if __name__ == '__main__':
    evaluate_completeness('2023-03-25')
