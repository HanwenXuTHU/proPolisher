import pandas as pd
import logging
import time

DICT_DEFAULT = {'TATAAT': 'TTGACA', 'AATTTA': 'TTACCA', 'GCGATA': 'CAATGG', 'TGGAAT': 'TGTGTA',
                'TCATCT': 'AAACGC', 'ACCTGG': 'GTTCCC', 'TCGGAT': 'TATCGA'}

def relation_reserve(csv_path, st=[46], ed=[52]):
    path = csv_path
    results = pd.read_csv(path)
    fakeB = list(results['fakeB'])
    realA = list(results['realA'])
    realB = list(results['realB'])
    n, cn = 0, 0
    for i in range(len(realB)):
        realAt, realBt, fakeBt = realA[i], realB[i], fakeB[i]
        for j in range(len(st)):
            for k in range(st[j], ed[j], 1):
                n = n + 1
                if fakeBt[k] == realBt[k]:
                    cn = cn + 1
    return cn/n


def polyAT_freq(valid_path, ref_path):
    A_dict_valid = {'AAAAA':0, 'AAAAAA':0, 'AAAAAAA': 0, 'AAAAAAAA': 0}
    A_dict_ref = {'AAAAA': 0, 'AAAAAA': 0, 'AAAAAAA': 0, 'AAAAAAAA': 0}
    T_dict_valid = {'TTTTT':0, 'TTTTTT':0, 'TTTTTTT': 0, 'TTTTTTTT': 0}
    T_dict_ref = {'TTTTT': 0, 'TTTTTT': 0, 'TTTTTTT': 0, 'TTTTTTTT': 0}
    valid_df = pd.read_csv(valid_path)
    ref_df = pd.read_csv(ref_path)
    fakeB = list(valid_df['fakeB'])
    realB = list(ref_df['realB'])
    for i in range(len(fakeB)):
        fakeBt = fakeB[i]
        for keys in A_dict_valid.keys():
            for j in range(0, len(fakeBt) - len(keys) + 1):
                if fakeBt[j : j + len(keys)] == keys:
                    A_dict_valid[keys] += 1
        for keys in T_dict_valid.keys():
            for j in range(0, len(fakeBt) - len(keys) + 1):
                if fakeBt[j : j + len(keys)] == keys:
                    T_dict_valid[keys] += 1

    for i in range(len(realB)):
        realBt = realB[i]
        for keys in A_dict_ref.keys():
            for j in range(0, len(realBt) - len(keys) + 1):
                if realBt[j : j + len(keys)] == keys:
                    A_dict_ref[keys] += 1
        for keys in T_dict_ref.keys():
            for j in range(0, len(realBt) - len(keys) + 1):
                if realBt[j : j + len(keys)] == keys:
                    T_dict_ref[keys] += 1

    for keys in A_dict_valid.keys():
        A_dict_valid[keys] = A_dict_valid[keys] / len(fakeB)
        A_dict_ref[keys] = A_dict_ref[keys] / len(realB)
    for keys in T_dict_valid.keys():
        T_dict_valid[keys] = T_dict_valid[keys] / len(fakeB)
        T_dict_ref[keys] = T_dict_ref[keys]/len(realB)

    return A_dict_valid, A_dict_ref, T_dict_valid, T_dict_ref


def get_logger():
    # 第一步，创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关
    # 第二步，创建一个handler，用于写入日志文件
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_path = 'cache/training_log/'
    log_name = log_path + rq + '.log'
    logfile = log_name
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
    # 第三步，定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    # 第四步，将logger添加到handler里面
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)  # 输出到console的log等级的开关
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

