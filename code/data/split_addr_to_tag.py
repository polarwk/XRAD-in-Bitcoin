import os

ransom_addr_file_loca = '../nfs/zhaoyu/ransom_addr'
all_ransoms = os.listdir(ransom_addr_file_loca)
for one_ransom in all_ransoms:
    one_ransom_addr = []
    with open('../cd_data/' + one_ransom[0:-17] + '_active.csv', 'a') as f1:
        for line in open(ransom_addr_file_loca + '/' + one_ransom, 'r'):
            line = line.split(',')
            addr = line[0]
            one_ransom_addr.append(addr)
        for line in open('../cd_data/ransom_addr_first_last_date.csv', 'r'):
            tmp = line.split(',')
            addr = tmp[0]
            if addr in one_ransom_addr:
                f1.write(line)
    f1.close()
