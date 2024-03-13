import os
import csv
from loguru import logger
csv.field_size_limit(1024*1024*1024)


def read_ransom_addr():

    ransom_addr_set = set()
    ransom_path = "../nfs/wk/TDSC/24440_addrs.csv"
    ransom_f = open(ransom_path, "r")
    ransom_reader = csv.reader(ransom_f)
    for row in ransom_reader:
        ransom_addr_set.add(row[0])
    
    return ransom_addr_set


def check_random_addr():
    ransom_addr_set = read_ransom_addr()

    pure_white_addr_path = "../nfs/wk/TDSC/pure_random_white_address/"

    random_addr_path = "../nfs/wk/TDSC/random_white_address/"
    for i in range(2013, 2021):
        logger.info(i)
        year_random_addr_parh = random_addr_path + "white_" + str(i) + ".csv"
        year_random_addr_f = open(year_random_addr_parh, "r")
        year_random_addr_reader = csv.reader(year_random_addr_f)

        year_pure_white_addr_path = pure_white_addr_path + "white_" + str(i) + ".csv"
        year_pure_white_addr_f = open(year_pure_white_addr_path, "a+")
        year_pure_white_addr_writer = csv.writer(year_pure_white_addr_f)

        for row in year_random_addr_reader:
            if row[0] in ransom_addr_set:
                logger.info("in")
            else:
                year_pure_white_addr_writer.writerow([row[0]])


def check_random_cascade_addr():
    ransom_addr_set = read_ransom_addr()

    pure_white_addr_path = "../nfs/wk/TDSC/pure_random_white_address/"

    random_addr_path = "../nfs/wk/TDSC/random_white_address/"
    for i in range(2013, 2021):
        logger.info(i)
        year_random_addr_parh = random_addr_path + "white_" + str(i) + ".csv"
        year_random_addr_f = open(year_random_addr_parh, "r")
        year_random_addr_reader = csv.reader(year_random_addr_f)

        year_pure_white_addr_path = pure_white_addr_path + "white_" + str(i) + ".csv"
        year_pure_white_addr_f = open(year_pure_white_addr_path, "a+")
        year_pure_white_addr_writer = csv.writer(year_pure_white_addr_f)

        for row in year_random_addr_reader:
            if row[0] in ransom_addr_set:
                logger.info("in")
            else:
                year_pure_white_addr_writer.writerow([row[0]])




if __name__ == "__main__":
    check_random_addr()
