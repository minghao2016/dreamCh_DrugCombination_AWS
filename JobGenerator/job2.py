def params(round_num, j1_type, excluded_feature_cnt):
    if round_num == 0:
        if j1_type == 'a':
            pastRound_bestC = 180
        else:
            pastRound_bestC = 140
    else:
        src_file_path = '/'.join(['data', str(round_num-1), 'J3condor', 'result', 'parameter.csv'])
        f = open(src_file_path)
        lines = f.read().split('\n')
        f.close()

        target_line = lines[1].strip()
        params = target_line.split(',')

        pastRound_bestC = int(float(params[0]))

    remain_feature_cnt = 954 - excluded_feature_cnt
    pbc = pastRound_bestC

    if remain_feature_cnt > 500 :
        c_list = [pbc - 200, pbc - 120, pbc - 60, pbc, pbc + 60, pbc + 120, pbc + 200]
    elif remain_feature_cnt > 300 :
        c_list = [pbc - 150, pbc - 80, pbc - 40, pbc, pbc + 40, pbc + 80, pbc + 150]
    elif remain_feature_cnt > 150:
        c_list = [pbc - 100, pbc - 50, pbc - 20, pbc, pbc + 20, pbc + 50, pbc + 100]
    elif remain_feature_cnt > 80:
        c_list = [pbc - 60, pbc - 30, pbc - 10, pbc, pbc + 10, pbc + 30, pbc + 60]
    else :
        c_list = [pbc - 40, pbc - 20, pbc - 10, pbc, pbc + 10, pbc + 20, pbc + 40]

    gamma_list = [0.061]
    #c_list = range(pastRound_bestC - 20*6, pastRound_bestC + 20*6, 20)
    #gamma_list = [x/1000.0 for x in range(1,100,20)]
    #gamma_list = [x/1000.0 for x in range(1,100,20)]

    values = [
            (c_value, gamma_value, dataset_num)
            #(c_value, 0.061, dataset_num)
            for c_value in c_list
            for gamma_value in gamma_list
            for dataset_num in range(0,10)
            if c_value > 1
            ]
    return values

if __name__ == '__main__':
    print len(params(0))
