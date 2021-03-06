import os
import json

all_features = [
        file_name.split('/')[0].split('_',1)[1].split('.')[0]
        for file_name in os.listdir('data/features/')]

def params(round_num, j1_type, include_features):
    if round_num == 0:
        if j1_type == 'a':
            pastRound_bestC = 100
        else:
            pastRound_bestC = 100
    else:
        src_file_path = '/'.join(['data', str(round_num-1), 'J6condor', 'parameter.csv'])
        f = open(src_file_path)
        lines = f.read().split('\n')
        f.close()

        target_line = lines[1].strip()
        params = target_line.split(',')

        pastRound_bestC = int(float(params[0]))

    # c values
    #c_list = range(50, 150, 25)
    #c_list = range(pastRound_bestC - 20*6, pastRound_bestC + 20*6, 20)
    c_list = range(pastRound_bestC - 10*3, pastRound_bestC + 10*3, 10)

    # gamma values
    gamma_list = [x/1000.0 for x in range(40,100,20)]

    # for test
    #c_list = range(60, 90, 10)
    #gamma_list = [x/1000.0 for x in range(1,100,25)]

    # feature candidates
    include_features = set(include_features)
    test_features = [feature
            for feature in all_features
            if feature not in include_features]

    if len(test_features) == 0:
        print "########################################"
        print "Done!\nNo remain feature group"
        print "########################################"
        sys.exit(0)


    # packing parameter and group candidates
    values = [
            (feature, c_value, gamma_value, set_num)
            for feature in test_features
            for c_value in c_list
            for gamma_value in gamma_list
            for set_num in range(0,10)
            if c_value > 1
            ]

    return values

if __name__ == '__main__':
    import pprint
    pprint.pprint(params(0, 'a', [
        'additionscore',
        'cellline',
        'drug',
        'einf',
        'maxconc',
        'mutation_frombest',
        'mutation_fromgdsc',
        'rule',
        'target']))
