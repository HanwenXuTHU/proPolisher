import pandas as pd

DICT_DEFAULT = {'TATAAT': 'TTGACA', 'AATTTA': 'TTACCA', 'GCGATA': 'CAATGG', 'TGGAAT': 'TGTGTA',
                'TCATCT': 'AAACGC', 'ACCTGG': 'GTTCCC', 'TCGGAT': 'TATCGA'}


def relation_reserve(csv_path, reserve_dict=DICT_DEFAULT, st=46, ed=52):
    path = csv_path
    results = pd.read_csv(path)
    fakeB = list(results['fakeB'])
    realB = list(results['realB'])
    n, cn = 0, 0
    for i in range(len(realB)):
        realBt, fakeBt = realB[i], fakeB[i]
        keys = []
        for k in range(len(realB)):
            if realBt[k] != 'M':
                keys.append(realBt[k])
        keys = ''.join(keys)
        for k in range(st, ed, 1):
            n = n + 1
            if fakeBt[k] == reserve_dict[keys][k - st]:
                cn = cn + 1
    return cn/n