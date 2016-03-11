def params(round_num):
    if round_num == 0:
        pastRound_bestC = 180
    else:
        src_file_path = '/'.join(['data', str(round_num-1), 'J3condor', 'result', 'parameter.csv'])
        f = open(src_file_path)
        lines = f.read().split('\n')
        f.close()

        target_line = lines[1].strip()
        params = target_line.split(',')

        pastRound_bestC = int(float(params[0]))


    c_list = range(pastRound_bestC - 20*6, pastRound_bestC + 20*6, 20)
    gamma_list = [x/1000.0 for x in range(1,100,20)]

    values = [
            (c_value, gamma_value, dataset_num)
            for c_value in c_list
            for gamma_value in gamma_list
            for dataset_num in range(0,10)
            if c_value > 1
            ]

    return values

if __name__ == '__main__':
    print len(params(0))
