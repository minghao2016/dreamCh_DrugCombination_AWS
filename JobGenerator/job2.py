import os


def params(round_num, j1_type, include_features):
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

    """ EXCLUD VER
    #remain_feature_cnt = 991  - excluded_feature_cnt
    remain_feature_cnt = 1298  - excluded_feature_cnt
    pbc = pastRound_bestC


    if remain_feature_cnt > 500 :
        c_list = [pbc-400,pbc-300, pbc - 200, pbc - 120, pbc - 90, pbc - 60, pbc - 30,  pbc, pbc + 30, pbc + 60, pbc + 90, pbc + 120, pbc + 200, pbc+300, pbc+400]
    elif remain_feature_cnt > 300 :
        c_list = [pbc-350,pbc-250, pbc - 150, pbc - 80, pbc - 60, pbc - 40, pbc - 20, pbc, pbc + 20, pbc + 40, pbc + 60, pbc + 80, pbc + 150, pbc+250, pbc+350]
    elif remain_feature_cnt > 150:
        c_list = [pbc-300,pbc-200,pbc - 100, pbc - 60, pbc-40, pbc - 20, pbc - 10, pbc, pbc + 10, pbc + 20, pbc+40, pbc + 60, pbc + 100, pbc+200, pbc+300]
    elif remain_feature_cnt > 80:
        c_list = [pbc-250, pbc-150, pbc - 60, pbc - 40, pbc-30, pbc - 20, pbc - 10 , pbc, pbc + 10, pbc + 20, pbc+30, pbc + 40, pbc + 60, pbc + 150, pbc+250]
    elif remain_feature_cnt <= 0 :
        c_list = range(60 , 400, 10)
    else :
        c_list = [pbc - 40, pbc - 20, pbc - 10, pbc, pbc + 10, pbc + 20, pbc + 40]
    """

    # c values
    c_list = range(60, 300, 10)

    # gamma values
    """
    if remain_feature_cnt <= 0 :
        gamma_list = [x/1000.0 for x in range(1,100,10)]
    else:
        gamma_list = [0.061]
    """
    c_list = range(pastRound_bestC - 20*6, pastRound_bestC + 20*6, 20)
    gamma_list = [x/1000.0 for x in range(1,100,20)]

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
    print len(params(0, 'a', ['DSS_median', 'target']))
