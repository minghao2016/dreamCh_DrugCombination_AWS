# -* coding:utf-8 *-
from multiprocessing import Process
from subprocess import call
import sys
import pandas as pd
import os, shutil
import Constant
import csv
#import email_sender


#input1:defaultTrainlibfm
#input2:defaultTestlibfm

default_feature_dir = '/'.join(['data', 'features']) + '/exclude_'

#defaultTestlibfm = "includeTest_single_2.libfm"
#defaultTrainlibfm = "includeTrain_single_2.libfm"
defaultTestlibfm = "tmpTest_single_1a_expanded.libfm"
defaultTrainlibfm = "tmpTrain_single_1a_expanded.libfm"
# defaultTrainlibfm = sys.argv[0]
# defaultTestlibfm = sys.argv[1]

"""
def single_loop(i, dataFilePath, droplines):
    currfilePath = dataFilePath+"set/"+str(i) + "/"
    basicTrainlibfm = currfilePath+defaultTrainlibfm
    basicTestlibfm = currfilePath+defaultTestlibfm

    print "loading..."
    originaltrainDF = Constant.libfmFileToDF(basicTrainlibfm)
    originaltestDF = Constant.libfmFileToDF(basicTestlibfm)

    newtrainDF = originaltrainDF.drop(originaltrainDF.columns[droplines], axis=1)
    newtestDF = originaltestDF.drop(originaltrainDF.columns[droplines], axis=1)

    Constant.makeLibfmFileWithDF(originaltrainDF, newtrainDF,currfilePath+"Train_single_new.libfm")
    Constant.makeLibfmFileWithDF(originaltestDF, newtestDF,currfilePath+"Test_single_new.libfm")
"""

def execute(round_num):
    round_num_str = str(round_num)

    include_features= set()
    include_ids     = set()

    # read initial default feature group
    f = open('data/initial_default_feature_group.txt')
    include_features.update(f.read().splitlines())
    f.close()

    print "-----------------------------------------"
    for round_idx in range(round_num):
    	j6FilePath = "data/" + str(round_idx) + "/J6condor/result/"

	#test
	J6FileNames = os.listdir(j6FilePath)

        for J6FileName in J6FileNames:
	    f = open(j6FilePath+J6FileName,"r")
            include_features.update(f.read().splitlines())
            f.close()

    for include_feature in include_features:
        feature_file_path = default_feature_dir + include_feature + '.csv'
        f = open(feature_file_path, 'r')
        csv_f = csv.reader(f)
        csv_f.next()
        for row in csv_f:
            include_ids.add(int(row[0]))
        f.close()

        """ EXCLUDE VERSION
        lines = list()
	for J6FileName in J6FileNames:
	    print j6FilePath+J6FileName
	    f = open(j6FilePath+J6FileName,"r")
            lines = map(int, f.read().splitlines())
            droplinesSet = droplinesSet.union(set(lines))
            f.close()

        if len(lines) == 0 and round_num != 0 and round_num == round_idx:
            email_sender.send("Finish!")
            sys.exit()
        """

    print include_features
    print len(include_ids)," features will be used"
    print "-----------------------------------------"

    curr_round_dir = "data/" + str(round_num)  + "/"
    try:
        os.makedirs(curr_round_dir)
    except OSError:
        pass
    except e:
        raise e


    # write default ids
    f = open(curr_round_dir + 'default_ids.txt', 'w')
    for include_id in include_ids:
        f.write(str(include_id))
        f.write('\n')
    f.close()

    # write default groups
    f = open(curr_round_dir + 'default_groups.txt', 'w')
    for include_feature in include_features:
        f.write(include_feature)
        f.write('\n')
    f.close()

    return include_features
    """ EXCLUDE VERSION
    processes = list()
    for i in range(0,10):
	p = Process(target=single_loop, args=(i,dataFilePath, droplines))
        processes.append(p)
	p.start()

    for p in processes:
        p.join()

    total_j6_file_path = "data/" + str(round_idx) + "/J6condor/result/excludedIndexes.txt"
    f = open(total_j6_file_path, 'w')
    f.write('\n'.join([ str(val) for val in droplines ]))
    f.close()


    tmp_excluded = open('excludedIndexes.txt', 'w')
    tmp_excluded.write('\n'.join([ str(val) for val in droplines ]))
    tmp_excluded.close()

    return droplinesSet
    """
    #`call(["bash", "uploader.sh"])

if __name__ == '__main__':
    execute(3)
