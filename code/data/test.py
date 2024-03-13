addr_tag = {}
for line in open('D://tifs//allAddresses.txt', 'r'):
    if '\n' in line:
        line = line[0:-1]
        tmp = line.split(',')
        tag = tmp[1] + '   ' + tmp[2]
        addr_tag[tmp[0]] = tag

res = {}
for line in open('D://tifs//24486_addrs_txs_nums.csv', 'r'):
    if '\n' in line:
        line = line[0:-1]
        line = line.split(',')
        addr = line[0]
        if addr in addr_tag:
            res[addr] = addr_tag[addr]

with open('D://tifs//addr_tag.csv', 'a') as f1:
    for i in res:
        f1.write(i + ',' + res[i] + '\n')


